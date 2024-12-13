# src/app.py
import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline
import pandas as pd
import plotly.express as px
import time
from datetime import datetime
import numpy as np
from typing import Dict, List
import asyncio
import os
from utils import download_nltk_data
from fact_checker import FactChecker
from source_checker import SourceChecker
from realtime_processor import RealTimeProcessor
from dotenv import load_dotenv
from dashboard import DashboardManager
from alert_system import AlertSystem
from integration_layer import (
    IntegrationLayer,
    APIIntegration,
    WebhookIntegration,
    SlackIntegration,
    EmailIntegration
)
import requests
from datetime import datetime

# Load environment variables and download NLTK data
load_dotenv()
download_nltk_data()

class MisinformationDetector:
    def __init__(self):
        # Initialize NLTK components with error handling
        try:
            self.punkt_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            self.word_tokenizer = nltk.WordPunctTokenizer()
            self.stop_words = set(stopwords.words('english'))
        except LookupError as e:
            print(f"Error loading NLTK resources: {e}")
            # Fallback initialization
            self.punkt_tokenizer = None
            self.word_tokenizer = None
            self.stop_words = set()
            
        self.classifier = pipeline("zero-shot-classification", 
                                 model="facebook/bart-large-mnli")
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.fact_checker = FactChecker()
        self.source_checker = SourceChecker()
        self.alert_system = AlertSystem()
        
        # Initialize integration layer
        self.integration_layer = IntegrationLayer()
        self.setup_integrations()
    
    def setup_integrations(self):
        """Setup integration providers"""
        # Example API integration
        api_integration = APIIntegration(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("API_BASE_URL")
        )
        self.integration_layer.register_provider("api", api_integration)
        
        # Example Slack integration
        slack_integration = SlackIntegration(
            slack_token=os.getenv("SLACK_TOKEN"),
            channel=os.getenv("SLACK_CHANNEL")
        )
        self.integration_layer.register_provider("slack", slack_integration)
        
        # Example webhook integration
        webhook_integration = WebhookIntegration(
            webhook_url=os.getenv("WEBHOOK_URL")
        )
        self.integration_layer.register_provider("webhook", webhook_integration)
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text into sentences"""
        if not text:
            return []
        
        try:
            if self.punkt_tokenizer:
                return self.punkt_tokenizer.tokenize(text)
            else:
                # Fallback to sent_tokenize
                return sent_tokenize(text)
        except Exception as e:
            print(f"Error in sentence tokenization: {e}")
            return [text]
    
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        try:
            # Use word tokenizer if available, otherwise fall back to word_tokenize
            tokens = (self.word_tokenizer.tokenize(text) if self.word_tokenizer 
                     else word_tokenize(text))
            
            basic_terms = [word for word in tokens 
                          if word.isalnum() 
                          and word.lower() not in self.stop_words
                          and len(word) > 2]
            
            try:
                pos_tags = nltk.pos_tag(basic_terms)
                key_terms = [word for word, pos in pos_tags 
                            if (pos.startswith('NN') or pos.startswith('JJ')) 
                            and word.lower() not in self.stop_words
                            and word.isalnum()]
                return list(set(key_terms))
            except Exception as e:
                print(f"POS tagging failed, using basic filtering: {e}")
                return list(set(basic_terms))
                
        except Exception as e:
            print(f"Error in extract_key_terms: {e}")
            return []

    def _extract_urls(self, text: str) -> List[str]:
        words = text.split()
        return [word for word in words if word.startswith(('http://', 'https://'))]

    def calculate_risk_score(self, classification_scores: List[float], 
                           fact_check_score: float, 
                           sentiment_score: Dict) -> float:
        misleading_score = classification_scores[2] if len(classification_scores) > 2 else 0
        sentiment_factor = 0.2 if sentiment_score['label'] == 'NEGATIVE' else 0
        
        risk_score = (misleading_score * 0.5 + 
                     (1 - fact_check_score) * 0.3 + 
                     sentiment_factor)
        
        return min(risk_score, 1.0)

    def analyze_text(self, text: str) -> List[Dict]:
        sentences = self.preprocess_text(text)
        results = []
        
        for sentence in sentences:
            classification = self.classifier(
                sentence,
                candidate_labels=[
                    "factual statement", 
                    "opinion", 
                    "misleading information"
                ]
            )
            
            sentiment = self.sentiment_analyzer(sentence)[0]
            fact_check_results = self.fact_checker.check_claim(sentence)
            urls = self._extract_urls(sentence)
            source_checks = [self.source_checker.check_source(url) for url in urls]
            
            risk_score = self.calculate_risk_score(
                classification['scores'],
                fact_check_results['credibility_score'],
                sentiment
            )
            
            key_terms = self.extract_key_terms(sentence)
            
            result = {
                'sentence': sentence,
                'classifications': classification['labels'],
                'classification_scores': classification['scores'],
                'sentiment': sentiment,
                'fact_check_results': fact_check_results,
                'source_checks': source_checks,
                'urls': urls,
                'risk_score': risk_score,
                'key_terms': key_terms,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Check for alerts
            alerts = self.alert_system.check_content(result)
            if alerts:
                for alert in alerts:
                    st.warning(f"⚠️ Alert: {alert.message}")
                    
                    # Try to send alert through integration
                    try:
                        alert_data = {
                            'message': alert.message,
                            'risk_score': alert.risk_score,
                            'timestamp': alert.timestamp,
                            'source_text': alert.source_text,
                            'severity': alert.severity,
                            'type': alert.type
                        }
                        self.integration_layer.send_alerts_sync(alert_data)
                    except Exception as e:
                        st.error(f"Failed to send alert: {str(e)}")
            
            results.append(result)
        
        return results

def create_analysis_charts(results: List[Dict]):
    df = pd.DataFrame(results)
    fig_timeline = px.line(
        df,
        x='timestamp',
        y='risk_score',
        title='Risk Score Timeline'
    )
    
    classifications = []
    scores = []
    for result in results:
        for cls, score in zip(result['classifications'], result['classification_scores']):
            classifications.append(cls)
            scores.append(score)
    
    fig_dist = px.bar(
        x=classifications,
        y=scores,
        title='Classification Distribution'
    )
    
    return fig_timeline, fig_dist

def integration_settings_tab():
    st.title("🔌 Integration Settings")
    
    # API Integration Settings
    st.header("API Integration")
    
    # Advanced API Configuration
    with st.expander("API Configuration", expanded=True):
        api_key = st.text_input("API Key", type="password")
        api_url = st.text_input("API Base URL")
        
        # Authentication Method
        auth_method = st.selectbox(
            "Authentication Method",
            ["API Key as Parameter", "Bearer Token", "Basic Auth", "No Auth"]
        )
        
        # Request Configuration
        col1, col2 = st.columns(2)
        with col1:
            request_method = st.selectbox("Request Method", ["GET", "POST", "PUT", "DELETE"])
            api_endpoint = st.text_input("Endpoint (e.g., /v1/test)", "/")
        with col2:
            param_key = st.text_input("API Key Parameter Name (if required)", "apiKey")
            content_type = st.selectbox("Content Type", [
                "application/json",
                "application/x-www-form-urlencoded",
                "multipart/form-data"
            ])
    
    # Test API button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Test API"):
            if api_url:  # Only URL is mandatory
                with st.spinner("Testing API connection..."):
                    try:
                        # Prepare headers
                        headers = {"Content-Type": content_type}
                        
                        # Handle different auth methods
                        params = {}
                        if auth_method == "API Key as Parameter":
                            params[param_key] = api_key
                        elif auth_method == "Bearer Token":
                            headers["Authorization"] = f"Bearer {api_key}"
                        elif auth_method == "Basic Auth":
                            headers["Authorization"] = f"Basic {api_key}"
                        
                        # Prepare URL
                        full_url = f"{api_url.rstrip('/')}{api_endpoint}"
                        
                        # Test data for POST/PUT requests
                        test_data = {
                            "message": "Test alert",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # Make request based on method
                        if request_method == "GET":
                            response = requests.get(
                                full_url,
                                headers=headers,
                                params=params
                            )
                        elif request_method == "POST":
                            response = requests.post(
                                full_url,
                                headers=headers,
                                params=params,
                                json=test_data if content_type == "application/json" else None,
                                data=test_data if content_type != "application/json" else None
                            )
                        elif request_method == "PUT":
                            response = requests.put(
                                full_url,
                                headers=headers,
                                params=params,
                                json=test_data if content_type == "application/json" else None,
                                data=test_data if content_type != "application/json" else None
                            )
                        else:  # DELETE
                            response = requests.delete(
                                full_url,
                                headers=headers,
                                params=params
                            )
                        
                        # Display response
                        if response.status_code in [200, 201, 202]:
                            st.success(f"✅ API connection successful! Status: {response.status_code}")
                            try:
                                st.json(response.json())
                            except:
                                st.text(response.text)
                        else:
                            st.error(f"❌ API test failed: {response.status_code}")
                            st.text(f"Response: {response.text}")
                            
                    except Exception as e:
                        st.error(f"❌ API test failed: {str(e)}")
            else:
                st.warning("Please enter API Base URL")
    
    # Slack Integration Settings
    st.header("Slack Integration")
    slack_token = st.text_input("Slack Token", type="password")
    slack_channel = st.text_input("Slack Channel")
    slack_message = st.text_area("Slack Message", "🔍 Test message from Misinformation Detector!")

    # Test Slack button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Send to Slack"):
            if slack_token and slack_channel and slack_message:
                with st.spinner("Sending message to Slack..."):
                    try:
                        headers = {
                            "Authorization": f"Bearer {slack_token}",
                            "Content-Type": "application/json"
                        }
                        message_data = {
                            "channel": slack_channel,
                            "text": slack_message
                        }
                        response = requests.post(
                            "https://slack.com/api/chat.postMessage",
                            headers=headers,
                            json=message_data
                        )
                        if response.ok:
                            st.success("✅ Slack message sent successfully!")
                        else:
                            st.error(f"❌ Slack send failed: {response.json().get('error', '')}")
                    except Exception as e:
                        st.error(f"❌ Slack send failed: {str(e)}")
            else:
                st.warning("Please enter Slack credentials and message")

    # Webhook Integration Settings
    st.header("Webhook Integration")
    webhook_url = st.text_input("Webhook URL")
    webhook_message = st.text_area("Webhook Message", "Test webhook alert")

    # Test Webhook button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Send to Webhook"):
            if webhook_url and webhook_message:
                with st.spinner("Sending webhook..."):
                    try:
                        test_data = {
                            "message": webhook_message,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        response = requests.post(webhook_url, json=test_data)
                        if response.ok:
                            st.success("✅ Webhook message sent successfully!")
                        else:
                            st.error(f"❌ Webhook send failed: {response.status_code}")
                    except Exception as e:
                        st.error(f"❌ Webhook send failed: {str(e)}")
            else:
                st.warning("Please enter webhook URL and message")

    # Save all settings
    if st.button("Save All Settings"):
        if any([api_key, api_url, slack_token, slack_channel, webhook_url]):
            # Save settings to session state or environment
            st.session_state.integration_settings = {
                "api": {"key": api_key, "url": api_url},
                "slack": {"token": slack_token, "channel": slack_channel},
                "webhook": {"url": webhook_url}
            }
            st.success("✅ All settings saved successfully!")
        else:
            st.warning("Please enter at least one integration setting")

def display_live_results(results: List[Dict]):
    if not results:
        st.warning("No results to display yet. Start monitoring to see analysis.")
        return

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    avg_risk = np.mean([r['risk_score'] for r in results])
    avg_fact_check = np.mean([r['fact_check_results']['credibility_score'] for r in results])
    avg_content_quality = np.mean([
        r['fact_check_results']['credibility_analysis']['components']['content_quality'] 
        for r in results
    ])
    
    with col1:
        st.metric("Average Risk Score", f"{avg_risk:.2%}")
    
    with col2:
        st.metric("Average Credibility Score", f"{avg_fact_check:.2%}")
    
    with col3:
        st.metric("Content Quality", f"{avg_content_quality:.2%}")
    
    with col4:
        high_risk_count = sum(1 for r in results if r['risk_score'] > 0.7)
        st.metric("High Risk Alerts", high_risk_count)

    # Timeline visualization
    if len(results) > 0:
        # Create multi-line chart for risk and credibility scores
        df = pd.DataFrame([{
            'timestamp': datetime.strptime(r['timestamp'], "%Y-%m-%d %H:%M:%S"),
            'risk_score': r['risk_score'],
            'credibility_score': r['fact_check_results']['credibility_score'],
            'text': r['sentence'][:50] + '...'
        } for r in results])
        
        fig_timeline = px.line(
            df,
            x='timestamp',
            y=['risk_score', 'credibility_score'],
            title='Risk and Credibility Score Timeline',
            hover_data=['text'],
            labels={'value': 'Score', 'variable': 'Metric'},
            color_discrete_map={
                'risk_score': 'red',
                'credibility_score': 'green'
            }
        )
        st.plotly_chart(fig_timeline)

    # Latest Analysis Section
    st.subheader("Latest Analyses")
    for result in results[-5:]:  # Show last 5 results
        with st.expander(f"Analysis for: {result['sentence'][:100]}..."):
            # Main scores
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.metric(
                    "Risk Score", 
                    f"{result['risk_score']:.2%}",
                    delta=None,
                    delta_color="inverse"
                )
            with col2:
                st.metric(
                    "Credibility Score",
                    f"{result['fact_check_results']['credibility_score']:.2%}"
                )
            with col3:
                sentiment_score = result['sentiment']['score']
                sentiment_label = result['sentiment']['label']
                st.metric(
                    "Sentiment",
                    f"{sentiment_label}",
                    delta=f"{sentiment_score:.2%}"
                )

            # Detailed Analysis
            st.markdown("### Detailed Analysis")
            
            # Classification breakdown
            st.write("**Classification Breakdown:**")
            for cls, score in zip(result['classifications'], 
                                result['classification_scores']):
                score_color = 'red' if cls == 'misleading information' and score > 0.5 else 'inherit'
                st.markdown(
                    f"- {cls}: <span style='color:{score_color}'>{score:.2%}</span>", 
                    unsafe_allow_html=True
                )

            # Credibility Components
            st.write("**Credibility Components:**")
            cred_cols = st.columns(4)
            components = result['fact_check_results']['credibility_analysis']['components']
            for i, (name, score) in enumerate(components.items()):
                with cred_cols[i]:
                    st.metric(
                        name.replace('_', ' ').title(),
                        f"{score:.2%}"
                    )

            # Warning Flags
            flags = result['fact_check_results']['credibility_analysis']['flags']
            if flags:
                st.warning("**Warning Flags:**")
                for flag in flags:
                    st.markdown(f"⚠️ {flag}")

            # Key Terms and Recommendations
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Key Terms:**")
                st.markdown(", ".join(result['key_terms']))
            
            with col2:
                recommendations = result['fact_check_results']['credibility_analysis']['analysis']['recommendations']
                if recommendations:
                    st.write("**Recommendations:**")
                    for rec in recommendations:
                        st.markdown(f"• {rec}")

            # Risk Level Indicator
            risk_level = result['fact_check_results']['credibility_analysis']['analysis']['risk_level']
            if risk_level == "High":
                st.error("🚨 High Risk Content Detected!")
            elif risk_level == "Medium":
                st.warning("⚠️ Medium Risk Content")
            else:
                st.success("✅ Low Risk Content")

            # Summary
            st.info(
                f"**Analysis Summary:** {result['fact_check_results']['credibility_analysis']['analysis']['summary']}"
            )

    # Add export functionality
    if len(results) > 0:
        st.download_button(
            label="Export Analysis Results",
            data=pd.DataFrame([{
                'timestamp': r['timestamp'],
                'text': r['sentence'],
                'risk_score': r['risk_score'],
                'credibility_score': r['fact_check_results']['credibility_score'],
                'risk_level': r['fact_check_results']['credibility_analysis']['analysis']['risk_level'],
                'summary': r['fact_check_results']['credibility_analysis']['analysis']['summary']
            } for r in results]).to_csv(index=False),
            file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )

async def process_live_feed(processor, placeholder, results):
    """Process live feed data"""
    for i in range(5):
        live_text = f"This is a simulated live feed message {i+1}. " \
                    f"{'Some misleading content.' if i % 2 else 'Factual content.'}"
        
        processor.add_text(live_text, source="Live Feed")
        await asyncio.sleep(2)
        
        # Update the display
        with placeholder.container():
            display_live_results(results)

def text_analysis_tab():
    st.title("🔍 Real-time Misinformation Detector")
    
    st.header("Input Text")
    text_input = st.text_area(
        "Enter text to analyze:",
        height=150
    )
    
    if st.button("Analyze Text"):
        if text_input:
            with st.spinner("Analyzing text..."):
                results = st.session_state.detector.analyze_text(text_input)
     
                # Store results in session state
                if 'monitoring_results' not in st.session_state:
                    st.session_state.monitoring_results = []
                st.session_state.monitoring_results.extend(results)
                           
                st.header("Analysis Results")
                
                for result in results:
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**Sentence:** {result['sentence']}")
                            st.markdown("**Classifications:**")
                            for cls, score in zip(result['classifications'], 
                                                result['classification_scores']):
                                st.markdown(f"- {cls}: {score:.2%}")
                            
                            st.markdown(f"**Key Terms:** {', '.join(result['key_terms'])}")
                            st.markdown(f"**Sentiment:** {result['sentiment']['label']} "
                                      f"({result['sentiment']['score']:.2%})")
                            
                            st.markdown("**Fact Check Results:**")
                            st.json(result['fact_check_results'])
                            
                            if result['urls']:
                                st.markdown("**Source Check Results:**")
                                for url, check in zip(result['urls'], result['source_checks']):
                                    st.markdown(f"URL: {url}")
                                    st.json(check)
                        
                        with col2:
                            st.metric("Risk Score", f"{result['risk_score']:.2%}")
                            st.metric("Fact-Check Score", 
                                    f"{result['fact_check_results']['credibility_score']:.2%}")
                        
                        if result['risk_score'] > 0.7:
                            st.warning("⚠️ High risk of misinformation!")
                        
                        st.divider()
                
                display_live_results(results)


def live_monitor_tab():
    st.title("📡 Live Broadcast Monitor")
    
    # Initialize monitoring results if not exists
    if 'monitoring_results' not in st.session_state:
        st.session_state.monitoring_results = []
    
    source_type = st.selectbox(
        "Select Source",
        ["Live News Feed", "Social Media Stream", "Custom Input"]
    )
    
    placeholder = st.empty()
    
    # Add monitoring controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        monitoring_speed = st.slider("Monitoring Speed (seconds)", 1, 10, 2)
    with col2:
        auto_refresh = st.checkbox("Auto Refresh", value=True)
    with col3:
        if st.button("Reset Monitoring"):
            st.session_state.monitoring_results = []
            st.session_state.detector.alert_system.alerts = []  # Clear alerts
            placeholder.empty()
            st.success("Monitoring reset successfully!")
            return
    
    if source_type == "Live News Feed":
        if st.button("Start Monitoring"):
            # Sample live feed data with potential misinformation
            live_feeds = [
                "Breaking news: Major technological breakthrough announced in renewable energy.",
                "ALERT: Unverified reports claim 5G towers cause health issues! #conspiracy",
                "Scientists discover new species in the Amazon rainforest.",
                "URGENT: Secret government documents reveal shocking conspiracy! Must share!",
                "Latest update: Stock market manipulation by AI algorithms, experts warn.",
                "BREAKING: Miracle cure for all diseases discovered! Buy now at discount!",
                "New study confirms climate change impacts on global weather patterns.",
                "EXPOSED: Deep state operatives controlling social media! Share before deleted!"
            ]
            
            with st.spinner("Monitoring live feed..."):
                for feed in live_feeds:
                    # Analyze the feed using existing detector
                    results = st.session_state.detector.analyze_text(feed)
                    st.session_state.monitoring_results.extend(results)
                    
                    # Update display
                    with placeholder.container():
                        # Show latest feed with timestamp
                        st.subheader("Latest Feed")
                        st.info(f"[{datetime.now().strftime('%H:%M:%S')}] {feed}")
                        
                        # Alert status
                        alerts = sum(1 for r in results if r['risk_score'] > 0.7)
                        if alerts > 0:
                            st.error(f"🚨 {alerts} high-risk content detected in this feed!")
                        
                        # Display results
                        display_live_results(st.session_state.monitoring_results)
                        
                        # Show monitoring statistics
                        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                        with stats_col1:
                            st.metric("Active Feeds", len(st.session_state.monitoring_results))
                        with stats_col2:
                            avg_risk = np.mean([r['risk_score'] for r in st.session_state.monitoring_results])
                            st.metric("Average Risk", f"{avg_risk:.2%}")
                        with stats_col3:
                            high_risk = sum(1 for r in st.session_state.monitoring_results if r['risk_score'] > 0.7)
                            st.metric("High Risk Alerts", high_risk)
                        with stats_col4:
                            total_alerts = len(st.session_state.detector.alert_system.alerts)
                            st.metric("Total Alerts", total_alerts)
                    
                    time.sleep(monitoring_speed)
    
    elif source_type == "Social Media Stream":
        if st.button("Start Monitoring"):
            # Sample social media posts with potential misinformation
            social_posts = [
                "Just heard that AI will replace all jobs by next year! #tech #future",
                "New study shows regular exercise improves mental health. #health #wellness",
                "SHOCKING: Celebrity reveals government mind control program! #conspiracy",
                "Beautiful sunset at the beach today! 🌅 #nature #peace",
                "This miracle product cures everything in 24 hours! DM for details! #health",
                "URGENT: Share this before they delete it! Hidden truth revealed! #wakeup",
                "Scientific study confirms benefits of meditation. #mindfulness",
                "EXPOSED: Secret society controlling world events! Must read! #truth"
            ]
            
            with st.spinner("Monitoring social media..."):
                for post in social_posts:
                    results = st.session_state.detector.analyze_text(post)
                    st.session_state.monitoring_results.extend(results)
                    
                    with placeholder.container():
                        # Show latest post with timestamp
                        st.subheader("Latest Social Media Post")
                        st.info(f"[{datetime.now().strftime('%H:%M:%S')}] {post}")
                        
                        # Alert status
                        alerts = sum(1 for r in results if r['risk_score'] > 0.7)
                        if alerts > 0:
                            st.error(f"🚨 {alerts} high-risk content detected in this post!")
                        
                        # Display results
                        display_live_results(st.session_state.monitoring_results)
                        
                        # Show hashtag analysis
                        hashtags = [tag for tag in post.split() if tag.startswith('#')]
                        if hashtags:
                            st.write("🏷️ Trending Hashtags:", ', '.join(hashtags))
                        
                        # Show monitoring statistics
                        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                        with stats_col1:
                            st.metric("Monitored Posts", len(st.session_state.monitoring_results))
                        with stats_col2:
                            avg_risk = np.mean([r['risk_score'] for r in st.session_state.monitoring_results])
                            st.metric("Average Risk", f"{avg_risk:.2%}")
                        with stats_col3:
                            high_risk = sum(1 for r in st.session_state.monitoring_results if r['risk_score'] > 0.7)
                            st.metric("High Risk Posts", high_risk)
                        with stats_col4:
                            total_alerts = len(st.session_state.detector.alert_system.alerts)
                            st.metric("Total Alerts", total_alerts)
                    
                    time.sleep(monitoring_speed)
    
    elif source_type == "Custom Input":
        custom_input = st.text_area("Enter custom text to monitor:")
        if st.button("Analyze Custom Input"):
            if custom_input:
                results = st.session_state.detector.analyze_text(custom_input)
                st.session_state.monitoring_results.extend(results)
                
                with placeholder.container():
                    # Show input with timestamp
                    st.subheader("Custom Input Analysis")
                    st.info(f"[{datetime.now().strftime('%H:%M:%S')}] {custom_input}")
                    
                    # Alert status
                    alerts = sum(1 for r in results if r['risk_score'] > 0.7)
                    if alerts > 0:
                        st.error(f"🚨 {alerts} high-risk content detected in this input!")
                    
                    # Display results
                    display_live_results(st.session_state.monitoring_results)
                    
                    # Show monitoring statistics
                    stats_col1, stats_col2 = st.columns(2)
                    with stats_col1:
                        high_risk = sum(1 for r in results if r['risk_score'] > 0.7)
                        st.metric("High Risk Content", high_risk)
                    with stats_col2:
                        total_alerts = len(st.session_state.detector.alert_system.alerts)
                        st.metric("Total Alerts", total_alerts)

    # Add export functionality
    if st.session_state.monitoring_results:
        st.download_button(
            label="📥 Export Monitoring Results",
            data=pd.DataFrame([{
                'timestamp': r['timestamp'],
                'text': r['sentence'],
                'risk_score': r['risk_score'],
                'credibility_score': r['fact_check_results']['credibility_score'],
                'alerts': sum(1 for a in st.session_state.detector.alert_system.alerts 
                            if a.source_text == r['sentence'])
            } for r in st.session_state.monitoring_results]).to_csv(index=False),
            file_name=f"monitoring_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )
        
def dashboard_tab():
    """Display the dashboard tab content"""
    st.session_state.dashboard_manager.display_dashboard(
        st.session_state.get('monitoring_results', [])
    )

def alerts_tab():
    """Display the alerts dashboard"""
    st.title("🚨 Alert Management System")
    
    # Add threshold controls at the top
    st.subheader("Alert Thresholds")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        new_risk = st.slider(
            "Risk Score Threshold",
            0.0, 1.0, 
            st.session_state.detector.alert_system.alert_thresholds['risk_score'],
            0.05
        )
    with col2:
        new_cred = st.slider(
            "Credibility Score Threshold",
            0.0, 1.0,
            st.session_state.detector.alert_system.alert_thresholds['credibility_score'],
            0.05
        )
    with col3:
        new_misleading = st.slider(
            "Misleading Content Threshold",
            0.0, 1.0,
            st.session_state.detector.alert_system.alert_thresholds['misleading_content'],
            0.05
        )
    
    if st.button("Update Thresholds"):
        st.session_state.detector.alert_system.alert_thresholds.update({
            'risk_score': new_risk,
            'credibility_score': new_cred,
            'misleading_content': new_misleading
        })
        st.success("✅ Thresholds updated successfully!")
    
    # Display the alerts dashboard
    st.session_state.detector.alert_system.display_alerts_dashboard()
    
def main():
    st.set_page_config(page_title="Real-time Misinformation Detector", 
                       layout="wide")
    
    # Initialize session state objects
    if 'detector' not in st.session_state:
        st.session_state.detector = MisinformationDetector()
    
    if 'dashboard_manager' not in st.session_state:
        st.session_state.dashboard_manager = DashboardManager()
    
    # Add sidebar navigation
    st.sidebar.title("Navigation")
    
    # Add logo/header to sidebar (optional)
    # st.sidebar.image("path_to_your_logo.png", width=100)  # Optional: Add your logo
    # st.sidebar.markdown("---")  # Add a separator
    
    # Create navigation menu
    nav_selection = st.sidebar.radio(
        "Go to",
        [
            "🔍 Text Analysis",
            "📡 Live Monitor",
            "📊 Dashboard",
            "🚨 Alerts",
            "🔌 Integrations"
        ]
    )
    
    # Add additional sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    st.sidebar.metric(
        "Active Alerts",
        len(st.session_state.detector.alert_system.alerts)
    )
    
    # Navigation logic
    if nav_selection == "🔍 Text Analysis":
        text_analysis_tab()
    
    elif nav_selection == "📡 Live Monitor":
        live_monitor_tab()
    
    elif nav_selection == "📊 Dashboard":
        dashboard_tab()
    
    elif nav_selection == "🚨 Alerts":
        alerts_tab()
    
    elif nav_selection == "🔌 Integrations":
        integration_settings_tab()
    
    # Add footer to sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        """
        This is a real-time misinformation detection system.
        Version: 1.0.0
        """
    )
    
    # Add help section to sidebar
    with st.sidebar.expander("Need Help?"):
        st.markdown("""
        **Quick Start Guide:**
        1. Use Text Analysis for single text analysis
        2. Use Live Monitor for real-time monitoring
        3. Check Dashboard for overall statistics
        4. Monitor Alerts for important notifications
        5. Configure Integrations for external connections
        """)

if __name__ == "__main__":
    main()