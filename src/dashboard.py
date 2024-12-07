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

    def _display_key_metrics(self, results: List[Dict]):
        """Display key performance metrics"""
        # Calculate metrics
        total_analyzed = len(results)
        high_risk_count = sum(1 for r in results if r['risk_score'] > 0.7)
        avg_credibility = np.mean([r['fact_check_results']['credibility_score'] for r in results])
        recent_trend = self._calculate_trend(results)

        # Display metrics in columns
        cols = st.columns(4)
        
        with cols[0]:
            st.metric(
                "Total Analyzed",
                total_analyzed,
                delta=f"+{len([r for r in results if (datetime.now() - datetime.strptime(r['timestamp'], '%Y-%m-%d %H:%M:%S')).seconds < 300])} in 5m"
            )
        
        with cols[1]:
            st.metric(
                "High Risk Content",
                high_risk_count,
                f"{(high_risk_count/total_analyzed)*100:.1f}%",
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
            alert_count = sum(1 for r in results 
                            if r['fact_check_results']['credibility_analysis']['flags'])
            st.metric(
                "Active Alerts",
                alert_count,
                f"{alert_count} requiring attention"
            )

    def _display_trend_analysis(self, results: List[Dict]):
        """Display trend analysis charts"""
        st.subheader("Trend Analysis")
        
        # Prepare data
        df = pd.DataFrame([{
            'timestamp': datetime.strptime(r['timestamp'], '%Y-%m-%d %H:%M:%S'),
            'risk_score': r['risk_score'],
            'credibility_score': r['fact_check_results']['credibility_score']
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

    def _display_risk_distribution(self, results: List[Dict]):
        """Display risk distribution chart"""
        st.subheader("Risk Distribution")
        
        # Calculate risk categories
        risk_categories = {
            'Low Risk': len([r for r in results if r['risk_score'] <= 0.4]),
            'Medium Risk': len([r for r in results if 0.4 < r['risk_score'] <= 0.7]),
            'High Risk': len([r for r in results if r['risk_score'] > 0.7])
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

    def _display_content_analysis(self, results: List[Dict]):
        """Display detailed content analysis"""
        st.subheader("Content Analysis")
        
        tabs = st.tabs(["Classification Analysis", "Source Analysis", "Key Terms"])
        
        with tabs[0]:
            self._display_classification_analysis(results)
        
        with tabs[1]:
            self._display_source_analysis(results)
        
        with tabs[2]:
            self._display_key_terms_analysis(results)

    def _display_classification_analysis(self, results: List[Dict]):
        """Display classification analysis"""
        # Aggregate classification scores
        classifications = {
            'factual': [],
            'opinion': [],
            'misleading': []
        }
        
        for result in results:
            for cls, score in zip(result['classifications'], 
                                result['classification_scores']):
                if 'factual' in cls.lower():
                    classifications['factual'].append(score)
                elif 'opinion' in cls.lower():
                    classifications['opinion'].append(score)
                elif 'misleading' in cls.lower():
                    classifications['misleading'].append(score)
        
        # Calculate averages
        avg_scores = {
            k: np.mean(v) if v else 0 
            for k, v in classifications.items()
        }
        
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

    def _display_source_analysis(self, results: List[Dict]):
        """Display source analysis"""
        # Extract and count sources
        sources = Counter()
        for result in results:
            analysis = result['fact_check_results']['credibility_analysis']
            if 'source' in analysis:
                sources[analysis['source']] += 1
        
        if sources:
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

    def _display_key_terms_analysis(self, results: List[Dict]):
        """Display key terms analysis"""
        # Aggregate key terms
        all_terms = Counter()
        for result in results:
            all_terms.update(result['key_terms'])
        
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

    def _display_alerts(self, results: List[Dict]):
        """Display real-time alerts"""
        st.subheader("⚠️ Real-Time Alerts")
        
        # Filter recent high-risk content
        recent_alerts = [
            r for r in results 
            if r['risk_score'] > 0.7 
            and (datetime.now() - datetime.strptime(r['timestamp'], '%Y-%m-%d %H:%M:%S')).seconds < 3600
        ]
        
        if recent_alerts:
            for alert in recent_alerts:
                with st.expander(f"High Risk Content Detected - {alert['timestamp']}"):
                    st.write(f"**Content:** {alert['sentence']}")
                    st.write(f"**Risk Score:** {alert['risk_score']:.2%}")
                    st.write("**Flags:**")
                    for flag in alert['fact_check_results']['credibility_analysis']['flags']:
                        st.warning(flag)
        else:
            st.success("No active alerts at this time")

    def _calculate_trend(self, results: List[Dict]) -> float:
        """Calculate recent trend in credibility scores"""
        if len(results) < 2:
            return 0.0
            
        recent_scores = [r['fact_check_results']['credibility_score'] 
                        for r in results[-10:]]
        if len(recent_scores) >= 2:
            return recent_scores[-1] - recent_scores[0]
        return 0.0
