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

        st.title("📊 Real-Time Misinformation Dashboard")

        # Top-level metrics
        self._display_key_metrics(results)
        
        # Charts row
        col1, col2 = st.columns(2)
        with col1:
            self._display_trend_analysis(results)
        with col2:
            self._display_risk_distribution(results)

        # Detailed analysis
        self._display_content_analysis(results)
        
        # Real-time alerts
        self._display_alerts(results)

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
    
    def _display_key_metrics(self, results: List[Dict]):
        """Display key performance metrics"""
        try:
            # Calculate metrics with safe access
            total_analyzed = len(results)
            
            # Safely access risk scores
            high_risk_count = sum(1 for r in results 
                                 if r.get('analysis', {}).get('risk_score', 0) > 0.7)
            
            # Safely access credibility scores
            credibility_scores = [
                r.get('analysis', {}).get('fact_check_results', {}).get('credibility_score', 0) 
                for r in results
            ]
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
                alert_count = sum(1 for r in results if r.get('analysis', {})
                                .get('fact_check_results', {})
                                .get('credibility_analysis', {})
                                .get('flags', []))
                st.metric(
                    "Active Alerts",
                    alert_count,
                    f"{alert_count} requiring attention"
                )
        except Exception as e:
            st.error(f"Error displaying metrics: {str(e)}")
    
    def _display_trend_analysis(self, results: List[Dict]):
        """Display trend analysis charts"""
        try:
            st.subheader("Trend Analysis")
            
            # Prepare data with safe access
            df = pd.DataFrame([{
                'timestamp': self._parse_timestamp(r.get('timestamp', '')),
                'risk_score': r.get('analysis', {}).get('risk_score', 0),
                'credibility_score': r.get('analysis', {})
                                  .get('fact_check_results', {})
                                  .get('credibility_score', 0)
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
    
    def _display_risk_distribution(self, results: List[Dict]):
        """Display risk distribution chart"""
        try:
            st.subheader("Risk Distribution")
            
            # Calculate risk categories with safe access
            risk_categories = {
                'Low Risk': len([r for r in results 
                               if r.get('analysis', {}).get('risk_score', 0) <= 0.4]),
                'Medium Risk': len([r for r in results 
                                  if 0.4 < r.get('analysis', {}).get('risk_score', 0) <= 0.7]),
                'High Risk': len([r for r in results 
                                if r.get('analysis', {}).get('risk_score', 0) > 0.7])
            }
            
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
        except Exception as e:
            st.error(f"Error displaying risk distribution: {str(e)}")

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
            # Aggregate classification scores
            classifications = {
                'factual': [],
                'opinion': [],
                'misleading': []
            }
            
            for result in results:
                analysis = result.get('analysis', {})
                if analysis:
                    # Get classifications and scores with safe access
                    cls_list = analysis.get('classifications', [])
                    score_list = analysis.get('classification_scores', [])
                    
                    # Safely zip the lists
                    for cls, score in zip(cls_list, score_list):
                        if 'factual' in cls.lower():
                            classifications['factual'].append(score)
                        elif 'opinion' in cls.lower():
                            classifications['opinion'].append(score)
                        elif 'misleading' in cls.lower():
                            classifications['misleading'].append(score)
            
            # Calculate averages with error handling
            avg_scores = {}
            for k, v in classifications.items():
                try:
                    avg_scores[k] = np.mean(v) if v else 0
                except Exception:
                    avg_scores[k] = 0
            
            if any(avg_scores.values()):  # Only create chart if we have data
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
    
    
    def _display_source_analysis(self, results: List[Dict]):
        """Display source analysis"""
        try:
            # Extract and count sources with safe access
            sources = Counter()
            for result in results:
                analysis = result.get('analysis', {})
                fact_check = analysis.get('fact_check_results', {})
                cred_analysis = fact_check.get('credibility_analysis', {})
                source = cred_analysis.get('source', 'Unknown')
                sources[source] += 1
            
            if sources and sources.keys() != {'Unknown'}:  # Check if we have meaningful data
                # Create bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(sources.keys()),
                        y=list(sources.values()),
                        marker_color=self.chart_colors['neutral']
                    )
                ])
                
                fig.update_layout(
                    title='Content Sources Distribution',
                    xaxis_title='Source',
                    yaxis_title='Count'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No source data available")
                
        except Exception as e:
            st.error(f"Error displaying source analysis: {str(e)}")
    
    
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