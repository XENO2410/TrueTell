# src/dashboard.py
from typing import Dict, List
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from collections import Counter

class DashboardManager:
    def __init__(self):
        self.chart_colors = {
            'risk': 'red',
            'credibility': 'green',
            'neutral': 'gray',
            'positive': '#00CC96',
            'negative': '#EF553B',
            'warning': '#FFB020'
        }

    def display_dashboard(self, results: List[Dict]):
        """Display the main dashboard"""
        if not results:
            st.warning("No data available for dashboard. Start monitoring to see analytics.")
            return
    
        # Debug logging
        # st.write("Debug: Number of results:", len(results))
        # if results:
        #     st.write("Debug: Sample result structure:", results[0])

        # st.title("📊 Real-Time Misinformation Dashboard")

        # Validate results
        valid_results = [r for r in results if self._validate_result(r)]
        if not valid_results:
            st.error("No valid results found. Please check the data structure.")
            return
    
        st.title("📊 Real-Time Misinformation Dashboard")
        
        # Display metrics with valid results
        self._display_key_metrics(valid_results)
        
        # Charts row
        col1, col2 = st.columns(2)
        with col1:
            self._display_trend_analysis(valid_results)
        with col2:
            self._display_risk_distribution(valid_results)
    
        # Detailed analysis
        self._display_content_analysis(valid_results)
        
        # Real-time alerts
        self._display_alerts(valid_results)

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse different timestamp formats"""
        try:
            # Try different timestamp formats
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S.%fZ'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue
                    
            # If no format matches, return current time
            return datetime.now()
        except Exception as e:
            print(f"Error parsing timestamp {timestamp_str}: {e}")
            return datetime.now()

    def _validate_result(self, result: Dict) -> bool:
        """Validate result structure"""
        try:
            # Check for required fields
            if 'analysis' in result:
                analysis = result['analysis']
                required_fields = ['classifications', 'classification_scores', 'risk_score']
                return all(field in analysis for field in required_fields)
            else:
                required_fields = ['classifications', 'classification_scores', 'risk_score']
                return all(field in result for field in required_fields)
        except Exception:
            return False
            
    def _display_key_metrics(self, results: List[Dict]):
        """Display key performance metrics"""
        try:
            # Calculate metrics with safe access
            total_analyzed = len(results)
            
            # Helper function to get risk score from either structure
            def get_risk_score(result):
                return (result.get('analysis', {}).get('risk_score', 0) 
                       if 'analysis' in result 
                       else result.get('risk_score', 0))
            
            # Helper function to get credibility score from either structure
            def get_credibility_score(result):
                if 'analysis' in result:
                    return (result['analysis']
                           .get('fact_check_results', {})
                           .get('credibility_score', 0))
                return (result.get('fact_check_results', {})
                       .get('credibility_score', 0))
            
            # Calculate high risk count
            high_risk_count = sum(1 for r in results if get_risk_score(r) > 0.7)
            
            # Calculate average credibility
            credibility_scores = [get_credibility_score(r) for r in results]
            avg_credibility = np.mean(credibility_scores) if credibility_scores else 0
            
            recent_trend = self._calculate_trend(results)
    
            # Display metrics in columns
            cols = st.columns(4)
            
            with cols[0]:
                recent_count = sum(1 for r in results if (
                    datetime.now() - self._parse_timestamp(r.get('timestamp', ''))).seconds < 300)
                st.metric(
                    "Total Analyzed",
                    total_analyzed,
                    delta=f"+{recent_count} in 5m"
                )
            
            with cols[1]:
                st.metric(
                    "High Risk Content",
                    high_risk_count,
                    f"{(high_risk_count/total_analyzed)*100:.1f}%" if total_analyzed > 0 else "0%",
                    delta_color="inverse"
                )
            
            with cols[2]:
                st.metric(
                    "Avg. Credibility",
                    f"{avg_credibility:.1%}",
                    f"{recent_trend:.1%}",
                    delta_color="normal"
                )
            
            with cols[3]:
                # Helper function to get flags
                def get_flags(result):
                    if 'analysis' in result:
                        return (result['analysis']
                               .get('fact_check_results', {})
                               .get('credibility_analysis', {})
                               .get('flags', []))
                    return (result.get('fact_check_results', {})
                           .get('credibility_analysis', {})
                           .get('flags', []))
                
                alert_count = sum(1 for r in results if get_flags(r))
                st.metric(
                    "Active Alerts",
                    alert_count,
                    f"{alert_count} requiring attention"
                )
                
        except Exception as e:
            st.error(f"Error displaying metrics: {str(e)}")
            st.exception(e)  # This will show the full traceback
    
    def _display_trend_analysis(self, results: List[Dict]):
        """Display trend analysis charts"""
        try:
            st.subheader("Trend Analysis")
            
            # Helper functions to get scores
            def get_risk_score(result):
                return (result.get('analysis', {}).get('risk_score', 0) 
                       if 'analysis' in result 
                       else result.get('risk_score', 0))
            
            def get_credibility_score(result):
                if 'analysis' in result:
                    return (result['analysis']
                           .get('fact_check_results', {})
                           .get('credibility_score', 0))
                return (result.get('fact_check_results', {})
                       .get('credibility_score', 0))
            
            # Prepare data with safe access
            df = pd.DataFrame([{
                'timestamp': self._parse_timestamp(r.get('timestamp', '')),
                'risk_score': get_risk_score(r),
                'credibility_score': get_credibility_score(r)
            } for r in results])
            
            # Create multi-line chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['risk_score'],
                name='Risk Score',
                line=dict(color=self.chart_colors['risk'])
            ))
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['credibility_score'],
                name='Credibility Score',
                line=dict(color=self.chart_colors['credibility'])
            ))
            
            fig.update_layout(
                title='Risk vs Credibility Trends',
                xaxis_title='Time',
                yaxis_title='Score',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying trend analysis: {str(e)}")
            st.exception(e)  # This will show the full traceback
    
    def _display_risk_distribution(self, results: List[Dict]):
        """Display risk distribution chart"""
        try:
            st.subheader("Risk Distribution")
            
            # Helper function to get risk score
            def get_risk_score(result):
                if 'analysis' in result:
                    return result['analysis'].get('risk_score', 0)
                return result.get('risk_score', 0)
            
            # Calculate risk categories
            risk_categories = {
                'Low Risk': len([r for r in results if get_risk_score(r) <= 0.4]),
                'Medium Risk': len([r for r in results if 0.4 < get_risk_score(r) <= 0.7]),
                'High Risk': len([r for r in results if get_risk_score(r) > 0.7])
            }
            
            # Debug logging
            st.write("Debug: Risk distribution:", risk_categories)
            
            if sum(risk_categories.values()) > 0:
                # Create pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=list(risk_categories.keys()),
                    values=list(risk_categories.values()),
                    marker=dict(colors=[
                        self.chart_colors['positive'],
                        self.chart_colors['warning'],
                        self.chart_colors['negative']
                    ])
                )])
                
                fig.update_layout(title='Content Risk Distribution')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No risk distribution data available")
                
        except Exception as e:
            st.error(f"Error displaying risk distribution: {str(e)}")
            st.exception(e)  # Show full traceback

    def _display_content_analysis(self, results: List[Dict]):
        """Display detailed content analysis"""
        try:
            st.subheader("Content Analysis")
            
            tabs = st.tabs(["Classification Analysis", "Source Analysis", "Key Terms"])
            
            with tabs[0]:
                self._display_classification_analysis(results)
            
            with tabs[1]:
                self._display_source_analysis(results)
            
            with tabs[2]:
                self._display_key_terms_analysis(results)
        except Exception as e:
            st.error(f"Error displaying content analysis: {str(e)}")
    
    def _display_classification_analysis(self, results: List[Dict]):
        """Display classification analysis"""
        try:
            # Debug logging
            # st.write("Debug: Processing classifications")
            
            # Aggregate classification scores
            classifications = {
                'factual': [],
                'opinion': [],
                'misleading': []
            }
            
            for result in results:
                # Try both data structures
                cls_list = []
                score_list = []
                
                # Direct access
                if 'classifications' in result and 'classification_scores' in result:
                    cls_list = result['classifications']
                    score_list = result['classification_scores']
                # Nested access
                elif 'analysis' in result:
                    analysis = result['analysis']
                    cls_list = analysis.get('classifications', [])
                    score_list = analysis.get('classification_scores', [])
                
                # Process classifications
                if cls_list and score_list:
                    for cls, score in zip(cls_list, score_list):
                        cls_lower = cls.lower()
                        if 'factual' in cls_lower:
                            classifications['factual'].append(score)
                        elif 'opinion' in cls_lower:
                            classifications['opinion'].append(score)
                        elif 'misleading' in cls_lower:
                            classifications['misleading'].append(score)
            
            # Calculate averages
            avg_scores = {}
            for k, v in classifications.items():
                avg_scores[k] = np.mean(v) if v else 0
            
            if any(avg_scores.values()):
                # Create bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(avg_scores.keys()),
                        y=list(avg_scores.values()),
                        marker_color=[
                            self.chart_colors['positive'],
                            self.chart_colors['neutral'],
                            self.chart_colors['negative']
                        ]
                    )
                ])
                
                fig.update_layout(
                    title='Average Classification Scores',
                    yaxis_title='Score',
                    yaxis_range=[0, 1]
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No classification data available")
                
        except Exception as e:
            st.error(f"Error displaying classification analysis: {str(e)}")
            st.exception(e)  # Show full traceback
    
    
    def _display_source_analysis(self, results: List[Dict]):
        """Display source analysis"""
        try:
            # Extract and count sources with multiple fallbacks
            sources = Counter()
            source_credibility = {}
            
            for result in results:
                # Try different ways to get source information
                source = None
                
                # Try direct source field
                if result.get('source'):
                    source = result['source']
                
                # Try source from fact check results
                elif result.get('fact_check_results', {}).get('credibility_analysis', {}).get('source'):
                    source = result['fact_check_results']['credibility_analysis']['source']
                
                # If we found a source, process it
                if source:
                    sources[source] += 1
                    
                    # Get credibility score with multiple fallbacks
                    credibility_score = 0.0
                    
                    # Try getting score from fact_check_results
                    if 'fact_check_results' in result:
                        # Try direct credibility score
                        credibility_score = result['fact_check_results'].get('credibility_score', 0.0)
                        
                        # If no direct score, try components
                        if credibility_score == 0.0:
                            components = result['fact_check_results'].get('credibility_analysis', {}).get('components', {})
                            source_cred = components.get('source_credibility', 0.0)
                            content_qual = components.get('content_quality', 0.0)
                            credibility_score = (source_cred + content_qual) / 2 if source_cred or content_qual else 0.0
                    
                    # Store score for this source
                    if source not in source_credibility:
                        source_credibility[source] = {'scores': [], 'count': 0}
                    source_credibility[source]['scores'].append(credibility_score)
                    source_credibility[source]['count'] += 1
                else:
                    # Add to "Unknown" category if no source found
                    sources['Unknown'] += 1
            
            # Create source analysis visualization
            if sources and set(sources.keys()) != {'Unknown'}:
                # Create bar chart for source distribution
                source_df = pd.DataFrame({
                    'Source': list(sources.keys()),
                    'Count': list(sources.values())
                }).sort_values('Count', ascending=False)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=source_df['Source'],
                        y=source_df['Count'],
                        marker_color=self.chart_colors['neutral'],
                        text=source_df['Count'],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title='Content Sources Distribution',
                    xaxis_title='Source',
                    yaxis_title='Count',
                    showlegend=False,
                    xaxis={'tickangle': 45}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display source credibility metrics
                st.subheader("Source Credibility Analysis")
                
                # Calculate number of rows needed (4 sources per row)
                num_sources = len(source_credibility)
                sources_per_row = 4
                
                # Process sources in groups of 4
                source_items = list(source_credibility.items())
                for i in range(0, num_sources, sources_per_row):
                    # Get the current group of sources (up to 4)
                    current_group = source_items[i:min(i + sources_per_row, num_sources)]
                    
                    # Create columns for this group
                    cols = st.columns(len(current_group))
                    
                    # Display metrics for each source in the group
                    for col, (source, data) in zip(cols, current_group):
                        with col:
                            avg_score = np.mean(data['scores']) if data['scores'] else 0.0
                            max_score = max(data['scores']) if data['scores'] else 0.0
                            
                            # Display metrics
                            st.metric(
                                f"{source}",
                                f"{avg_score:.1%}",
                                delta=f"{data['count']} articles",
                                help=f"""
                                Average Credibility: {avg_score:.1%}
                                Highest Credibility: {max_score:.1%}
                                Total Articles: {data['count']}
                                """
                            )
                            
                            # Add a small bar to visualize the credibility range
                            if data['scores']:
                                st.progress(avg_score)
                    
                    # Add separator between rows
                    if i + sources_per_row < num_sources:
                        st.write("---")
                
                # Add overall source quality metrics
                st.subheader("Overall Source Quality Metrics")
                metric_cols = st.columns(3)
                
                with metric_cols[0]:
                    avg_overall = np.mean([np.mean(d['scores']) for d in source_credibility.values() if d['scores']])
                    st.metric("Average Source Quality", f"{avg_overall:.1%}")
                
                with metric_cols[1]:
                    high_cred_sources = sum(1 for d in source_credibility.values() 
                                          if np.mean(d['scores']) > 0.7)
                    st.metric("High Credibility Sources", high_cred_sources)
                
                with metric_cols[2]:
                    total_articles = sum(d['count'] for d in source_credibility.values())
                    st.metric("Total Articles Analyzed", total_articles)
                
            else:
                # If no source data, show helpful message
                st.info("""
                    No detailed source data available. This could be because:
                    - Content doesn't contain source information
                    - Sources haven't been verified yet
                    - Content is from direct input
                    
                    Try analyzing content with source URLs for more detailed analysis.
                """)
                
                # Show source verification tips
                with st.expander("📚 Source Verification Tips"):
                    st.markdown("""
                        To get source analysis:
                        1. Include URLs in your content
                        2. Reference known sources
                        3. Add source metadata when submitting content
                        4. Enable source verification in settings
                    """)
                    
        except Exception as e:
            st.error(f"Error displaying source analysis: {str(e)}")
            st.exception(e)  # Show full traceback for debugging
    
    
    def _display_key_terms_analysis(self, results: List[Dict]):
        """Display key terms analysis"""
        try:
            # Aggregate key terms with safe access
            all_terms = Counter()
            for result in results:
                terms = result.get('analysis', {}).get('key_terms', [])
                if isinstance(terms, list):  # Verify we have a valid list
                    all_terms.update(terms)
            
            if all_terms:
                # Create word cloud or bar chart
                top_terms = dict(all_terms.most_common(10))
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(top_terms.keys()),
                        y=list(top_terms.values()),
                        marker_color=self.chart_colors['neutral']
                    )
                ])
                
                fig.update_layout(
                    title='Top Key Terms',
                    xaxis_title='Term',
                    yaxis_title='Frequency'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No key terms data available")
                
        except Exception as e:
            st.error(f"Error displaying key terms analysis: {str(e)}")
    
    def _display_alerts(self, results: List[Dict]):
        """Display real-time alerts"""
        try:
            st.subheader("⚠️ Real-Time Alerts")
            
            # Filter recent high-risk content with safe access
            recent_alerts = [
                r for r in results 
                if r.get('analysis', {}).get('risk_score', 0) > 0.7 
                and (datetime.now() - self._parse_timestamp(r.get('timestamp', ''))).seconds < 3600
            ]
            
            if recent_alerts:
                for alert in recent_alerts:
                    with st.expander(f"High Risk Content Detected - {alert.get('timestamp', 'Unknown time')}"):
                        st.write(f"**Content:** {alert.get('text', alert.get('sentence', 'No content available'))}")
                        st.write(f"**Risk Score:** {alert.get('analysis', {}).get('risk_score', 0):.2%}")
                        st.write("**Flags:**")
                        flags = (alert.get('analysis', {})
                               .get('fact_check_results', {})
                               .get('credibility_analysis', {})
                               .get('flags', []))
                        if flags:
                            for flag in flags:
                                st.warning(flag)
                        else:
                            st.info("No specific flags raised")
            else:
                st.success("No active alerts at this time")
                
        except Exception as e:
            st.error(f"Error displaying alerts: {str(e)}")

    def _calculate_trend(self, results: List[Dict]) -> float:
        """Calculate recent trend in credibility scores"""
        try:
            if len(results) < 2:
                return 0.0
                
            recent_scores = [
                r.get('analysis', {})
                .get('fact_check_results', {})
                .get('credibility_score', 0) 
                for r in results[-10:]
            ]
            if len(recent_scores) >= 2:
                return recent_scores[-1] - recent_scores[0]
            return 0.0
        except Exception as e:
            print(f"Error calculating trend: {str(e)}")
            return 0.0