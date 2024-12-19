# src/alert_system.py
from typing import Dict, List, Optional
from datetime import datetime
import streamlit as st
from dataclasses import dataclass
import json
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Alert:
    id: str
    timestamp: str
    severity: str  # 'HIGH', 'MEDIUM', 'LOW'
    type: str  # 'MISINFORMATION', 'CREDIBILITY', 'SENTIMENT', etc.
    message: str
    source_text: str
    risk_score: float
    metadata: Dict
    status: str  # 'NEW', 'ACKNOWLEDGED', 'RESOLVED'

class AlertSystem:
    def __init__(self):
        self.alerts = []
        self.alert_thresholds = {
            'risk_score': 0.7,
            'credibility_score': 0.3,
            'sentiment_threshold': -0.7,
            'misleading_content': 0.8
        }
        # Get integration settings from environment variables
        self.webhook_url = os.getenv('WEBHOOK_URL')
        self.slack_token = os.getenv('SLACK_TOKEN')
        self.slack_channel = os.getenv('SLACK_CHANNEL')
        self.load_alerts()

    def send_to_webhook(self, alert: Alert) -> bool:
        """Send alert to webhook"""
        try:
            # Format the alert data similar to the working integration
            payload = {
                "message": alert.message,
                "risk_score": alert.risk_score,
                "timestamp": alert.timestamp,
                "severity": alert.severity,
                "type": alert.type,
                "source_text": alert.source_text,
                "metadata": alert.metadata,
                "status": alert.status,
                # Add any additional fields you want to include
                "alert_id": alert.id
            }
            
            # Send the request without any special headers
            response = requests.post(
                self.webhook_url,
                json=payload
            )
            
            # Check if request was successful (status code 200)
            if response.ok:
                st.success("✅ Sent to Webhook successfully!")
                return True
            else:
                st.error(f"Webhook error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            st.error(f"Error sending to webhook: {str(e)}")
            return False

    def send_to_slack(self, alert: Alert) -> bool:
        """Send alert to Slack"""
        try:
            # Prepare Slack message with blocks for better formatting
            message = {
                "channel": self.slack_channel,
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"🚨 Alert: {alert.type}"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Severity:* {alert.severity}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Risk Score:* {alert.risk_score:.2%}"
                            }
                        ]
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Message:*\n{alert.message}"
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Source Text:*\n{alert.source_text}"
                        }
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": f"Alert ID: {alert.id} | Time: {alert.timestamp}"
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(
                'https://slack.com/api/chat.postMessage',
                headers={
                    'Authorization': f'Bearer {self.slack_token}',
                    'Content-Type': 'application/json'
                },
                json=message
            )
            
            if response.ok:
                slack_data = response.json()
                if slack_data.get('ok', False):
                    return True
                else:
                    st.error(f"Slack API error: {slack_data.get('error', 'Unknown error')}")
                    return False
            else:
                st.error(f"Slack error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            st.error(f"Error sending to Slack: {str(e)}")
            return False

    # Keep all your existing methods unchanged
    def load_alerts(self):
        """Load alerts from storage"""
        try:
            if os.path.exists('data/alerts.json'):
                with open('data/alerts.json', 'r') as f:
                    alerts_data = json.load(f)
                    self.alerts = [Alert(**alert) for alert in alerts_data]
        except Exception as e:
            st.error(f"Error loading alerts: {e}")
            self.alerts = []

    def save_alerts(self):
        """Save alerts to storage"""
        try:
            os.makedirs('data', exist_ok=True)
            with open('data/alerts.json', 'w') as f:
                json.dump([alert.__dict__ for alert in self.alerts], f)
        except Exception as e:
            st.error(f"Error saving alerts: {e}")

    def generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        return f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.alerts)}"

    def check_content(self, analysis_result: Dict) -> Optional[Alert]:
        """Check content for alert conditions"""
        alerts = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Check risk score
        if analysis_result['risk_score'] >= self.alert_thresholds['risk_score']:
            alerts.append(Alert(
                id=self.generate_alert_id(),
                timestamp=timestamp,
                severity='HIGH',
                type='RISK_SCORE',
                message=f"High risk content detected (Score: {analysis_result['risk_score']:.2%})",
                source_text=analysis_result['sentence'],
                risk_score=analysis_result['risk_score'],
                metadata={
                    'classifications': analysis_result['classifications'],
                    'scores': analysis_result['classification_scores']
                },
                status='NEW'
            ))

        # Check credibility
        credibility_score = analysis_result['fact_check_results']['credibility_score']
        if credibility_score <= self.alert_thresholds['credibility_score']:
            alerts.append(Alert(
                id=self.generate_alert_id(),
                timestamp=timestamp,
                severity='HIGH',
                type='CREDIBILITY',
                message=f"Low credibility content detected (Score: {credibility_score:.2%})",
                source_text=analysis_result['sentence'],
                risk_score=analysis_result['risk_score'],
                metadata={
                    'credibility_score': credibility_score,
                    'fact_check_results': analysis_result['fact_check_results']
                },
                status='NEW'
            ))

        # Check for misleading content
        for cls, score in zip(analysis_result['classifications'], 
                            analysis_result['classification_scores']):
            if 'misleading' in cls.lower() and score >= self.alert_thresholds['misleading_content']:
                alerts.append(Alert(
                    id=self.generate_alert_id(),
                    timestamp=timestamp,
                    severity='HIGH',
                    type='MISLEADING_CONTENT',
                    message=f"Highly misleading content detected (Score: {score:.2%})",
                    source_text=analysis_result['sentence'],
                    risk_score=score,
                    metadata={
                        'classification': cls,
                        'score': score
                    },
                    status='NEW'
                ))

        # Add alerts to the system
        self.alerts.extend(alerts)
        self.save_alerts()
        return alerts

    def get_active_alerts(self) -> List[Alert]:
        """Get all active (non-resolved) alerts"""
        return [alert for alert in self.alerts if alert.status != 'RESOLVED']

    def get_alerts_by_severity(self, severity: str) -> List[Alert]:
        """Get alerts filtered by severity"""
        return [alert for alert in self.alerts if alert.severity == severity]

    def update_alert_status(self, alert_id: str, new_status: str) -> bool:
        """Update the status of an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.status = new_status
                self.save_alerts()
                return True
        return False

    def display_alerts_dashboard(self):
        """Display alerts dashboard in Streamlit"""
        st.subheader("🚨 Alert Dashboard")

        # Alert statistics
        total_alerts = len(self.alerts)
        active_alerts = len(self.get_active_alerts())
        high_severity = len(self.get_alerts_by_severity('HIGH'))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Alerts", total_alerts)
        with col2:
            st.metric("Active Alerts", active_alerts)
        with col3:
            st.metric("High Severity", high_severity)

        # Alert filters
        st.subheader("Alert Filters")
        col1, col2 = st.columns(2)
        with col1:
            severity_filter = st.multiselect(
                "Filter by Severity",
                options=['HIGH', 'MEDIUM', 'LOW'],
                default=['HIGH']
            )
        with col2:
            status_filter = st.multiselect(
                "Filter by Status",
                options=['NEW', 'ACKNOWLEDGED', 'RESOLVED'],
                default=['NEW', 'ACKNOWLEDGED']
            )

        # Display filtered alerts
        filtered_alerts = [
            alert for alert in self.alerts
            if alert.severity in severity_filter
            and alert.status in status_filter
        ]

        if filtered_alerts:
            for idx, alert in enumerate(filtered_alerts):
                with st.expander(f"{alert.type}: {alert.message} ({alert.timestamp})"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write("**Source Text:**", alert.source_text)
                        st.write("**Details:**")
                        st.json(alert.metadata)
                    
                    with col2:
                        st.write("**Status:**", alert.status)
                        
                        # Integration buttons
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("📤 Webhook", key=f"webhook_{alert.id}_{idx}"):
                                if self.send_to_webhook(alert):
                                    st.success("✅ Sent!")
                        with col_b:
                            if st.button("💬 Slack", key=f"slack_{alert.id}_{idx}"):
                                if self.send_to_slack(alert):
                                    st.success("✅ Sent!")
                        
                        # Status buttons
                        if alert.status != 'RESOLVED':
                            if st.button("Mark Resolved", key=f"resolve_{alert.id}_{idx}"):
                                self.update_alert_status(alert.id, 'RESOLVED')
                                st.rerun()
                            
                            if alert.status == 'NEW':
                                if st.button("Acknowledge", key=f"ack_{alert.id}_{idx}"):
                                    self.update_alert_status(alert.id, 'ACKNOWLEDGED')
                                    st.rerun()
        else:
            st.info("No alerts match the current filters.")

        # Alert settings
        with st.expander("Alert Settings"):
            st.subheader("Alert Thresholds")
            col1, col2 = st.columns(2)
            
            with col1:
                new_risk_threshold = st.slider(
                    "Risk Score Threshold",
                    0.0, 1.0,
                    self.alert_thresholds['risk_score']
                )
                new_credibility_threshold = st.slider(
                    "Credibility Score Threshold",
                    0.0, 1.0,
                    self.alert_thresholds['credibility_score']
                )
            
            with col2:
                new_misleading_threshold = st.slider(
                    "Misleading Content Threshold",
                    0.0, 1.0,
                    self.alert_thresholds['misleading_content']
                )

            if st.button("Update Thresholds", key="update_thresholds"):
                self.alert_thresholds.update({
                    'risk_score': new_risk_threshold,
                    'credibility_score': new_credibility_threshold,
                    'misleading_content': new_misleading_threshold
                })
                st.success("Alert thresholds updated successfully!")