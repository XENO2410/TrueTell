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
from knowledge_graph import KnowledgeGraph
import json

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
        self.knowledge_graph = KnowledgeGraph()
        
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
    
    def update_knowledge_graph(self, results: List[Dict]):
        """Update knowledge graph with new results"""
        for result in results:
            try:
                # Prepare content for knowledge graph
                content = {
                    'text': result['sentence'],
                    'timestamp': result['timestamp'],
                    'credibility_score': result['fact_check_results']['credibility_score'],
                    'verified': False,
                    'classification': {
                        'labels': result['classifications'],
                        'scores': result['classification_scores']
                    },
                    'sentiment': result['sentiment']['label'],
                    'key_terms': result['key_terms'],
                    'risk_score': result['risk_score']
                }
                
                # Add sources if available
                if result['urls'] and result['source_checks']:
                    content['sources'] = [
                        {
                            'url': url,
                            'check_result': check
                        } for url, check in zip(result['urls'], result['source_checks'])
                    ]
                
                # Add to knowledge graph
                self.knowledge_graph.add_content(content)
                
            except Exception as e:
                print(f"Error updating knowledge graph: {e}")
            
    def analyze_text(self, text: str) -> List[Dict]:
        """
        Analyze text with integrated knowledge graph support
        """
        sentences = self.preprocess_text(text)
        results = []
        
        for sentence in sentences:
            try:
                # Existing classification
                classification = self.classifier(
                    sentence,
                    candidate_labels=[
                        "factual statement", 
                        "opinion", 
                        "misleading information"
                    ]
                )
                
                # Existing analysis components
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
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Construct result dictionary
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
                    'timestamp': timestamp
                }
                
                # Add to knowledge graph
                try:
                    knowledge_graph_content = {
                        'text': sentence,
                        'timestamp': timestamp,
                        'credibility_score': fact_check_results['credibility_score'],
                        'verified': False,
                        'classification': classification,
                        'sentiment': sentiment['label'],
                        'key_terms': key_terms,
                        'risk_score': risk_score
                    }
                    
                    # Add sources if available
                    if urls and source_checks:
                        knowledge_graph_content['sources'] = [
                            {
                                'url': url,
                                'check_result': check
                            } for url, check in zip(urls, source_checks)
                        ]
                    
                    # Add to knowledge graph
                    self.knowledge_graph.add_content(knowledge_graph_content)
                    
                    # Get context from knowledge graph
                    graph_context = self.knowledge_graph.get_claim_context(
                        f"claim_{timestamp.replace(' ', '_').replace(':', '')}"
                    )
                    
                    # Add graph context to result
                    result['knowledge_graph_context'] = {
                        'related_claims': graph_context['related_claims'],
                        'related_entities': graph_context['entities'],
                        'sources': graph_context['sources']
                    }
                    
                except Exception as e:
                    print(f"Knowledge graph integration error: {e}")
                    result['knowledge_graph_context'] = {
                        'error': str(e),
                        'related_claims': [],
                        'related_entities': [],
                        'sources': []
                    }
                
                # Check for alerts
                alerts = self.alert_system.check_content(result)
                if alerts:
                    for alert in alerts:
                        st.warning(f"⚠️ Alert: {alert.message}")
                        
                        try:
                            # Enhanced alert data with knowledge graph context
                            alert_data = {
                                'message': alert.message,
                                'risk_score': alert.risk_score,
                                'timestamp': alert.timestamp,
                                'source_text': alert.source_text,
                                'severity': alert.severity,
                                'type': alert.type,
                                'related_claims': result['knowledge_graph_context']['related_claims'],
                                'related_entities': result['knowledge_graph_context']['related_entities']
                            }
                            self.integration_layer.send_alerts_sync(alert_data)
                        except Exception as e:
                            st.error(f"Failed to send alert: {str(e)}")
                
                results.append(result)
                
            except Exception as e:
                print(f"Error analyzing sentence: {e}")
                # Add error result
                results.append({
                    'sentence': sentence,
                    'error': str(e),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        
        # After processing all sentences, analyze patterns
        try:
            if results:
                patterns = self.knowledge_graph.analyze_claim_patterns()
                # Add pattern analysis to the last result
                results[-1]['pattern_analysis'] = patterns
        except Exception as e:
            print(f"Error analyzing patterns: {e}")
        
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

def knowledge_graph_tab():
    st.title("🕸️ Knowledge Graph Analysis")
    
    # Add sidebar controls for Export/Import
    st.sidebar.markdown("### Knowledge Graph Controls")
    
    # Export/Import section
    with st.sidebar.expander("Export/Import"):
        col1, col2 = st.columns(2)
        
        # Export functionality
        with col1:
            if st.button("Export Graph"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"knowledge_graph_{timestamp}.json"
                try:
                    st.session_state.detector.knowledge_graph.export_graph(filename)
                    st.success(f"✅ Graph exported to {filename}")
                    
                    # Add download button for the exported file
                    with open(filename, 'r') as f:
                        st.download_button(
                            label="Download Graph File",
                            data=f.read(),
                            file_name=filename,
                            mime="application/json"
                        )
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")
        
        # Import functionality
        with col2:
            uploaded_file = st.file_uploader("Import Graph", type="json")
            if uploaded_file is not None:
                try:
                    # Create a temporary file to handle the upload
                    with open("temp_graph.json", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.session_state.detector.knowledge_graph.import_graph("temp_graph.json")
                    st.success("✅ Graph imported successfully!")
                except Exception as e:
                    st.error(f"❌ Import failed: {str(e)}")

    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["Graph View", "Pattern Analysis", "Entity Explorer"])
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Graph Visualization")
            st.session_state.detector.knowledge_graph.visualize()
            
            # Add refresh button with proper rerun command
            if st.button("🔄 Refresh Visualization"):
                st.rerun()  # Updated from experimental_rerun
        
        with col2:
            st.subheader("Graph Statistics")
            stats = st.session_state.detector.knowledge_graph.get_statistics()
            
            # Display metrics without keys
            st.metric("🔵 Total Nodes", stats['total_nodes'])
            st.metric("🔗 Total Relationships", stats['total_edges'])
            
            # Display node type distribution
            if 'node_types' in stats:
                st.write("**Node Type Distribution:**")
                for node_type, count in stats['node_types'].items():
                    st.write(f"- {node_type}: {count}")
            
    with tab2:
        st.subheader("Pattern Analysis")
        
        # Add analysis options
        analysis_type = st.radio(
            "Select Analysis Type",
            ["Entity Co-occurrence", "Temporal Patterns", "Narrative Chains"],
            horizontal=True
        )
        
        if st.button("Analyze Patterns"):  # Removed key parameter
            with st.spinner("Analyzing patterns..."):
                patterns = st.session_state.detector.knowledge_graph.analyze_claim_patterns()
                
                if analysis_type == "Entity Co-occurrence":
                    st.write("**Entity Co-occurrence Analysis**")
                    if patterns['common_entities']:
                        # Sort entities by frequency
                        sorted_entities = dict(sorted(
                            patterns['common_entities'].items(),
                            key=lambda x: x[1],
                            reverse=True
                        ))
                        
                        fig = px.bar(
                            x=list(sorted_entities.keys())[:10],
                            y=list(sorted_entities.values())[:10],
                            title="Most Common Entities",
                            labels={'x': 'Entity', 'y': 'Frequency'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No entity co-occurrence data available.")
                
                elif analysis_type == "Temporal Patterns":
                    st.write("**Temporal Distribution Analysis**")
                    if patterns['temporal_patterns']:
                        fig = px.line(
                            x=list(patterns['temporal_patterns'].keys()),
                            y=list(patterns['temporal_patterns'].values()),
                            title="Content Distribution Over Time",
                            labels={'x': 'Hour', 'y': 'Number of Claims'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No temporal pattern data available.")
                
                else:  # Narrative Chains
                    st.write("**Narrative Chain Analysis**")
                    if patterns['narrative_chains']:
                        for idx, chain in enumerate(patterns['narrative_chains']):
                            with st.expander(f"Narrative Chain {idx+1} ({len(chain)} claims)"):
                                for claim in chain:
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.write(claim['text'])
                                    with col2:
                                        st.progress(claim['credibility_score'])
                                        st.caption(f"Credibility: {claim['credibility_score']:.2%}")
                    else:
                        st.info("No narrative chains found.")
    
    with tab3:
        st.subheader("Entity Explorer")
        
        # Add search functionality
        search_query = st.text_input("🔍 Search Entities", "")
        
        # Entity type filter
        entity_types = st.multiselect(
            "Filter by Entity Type",
            options=list(st.session_state.detector.knowledge_graph.entity_types),
            default=["PERSON", "ORG"]
        )
        
        # Display entities
        if entity_types:
            if search_query:
                # Search within filtered entities
                entities = st.session_state.detector.knowledge_graph.search_entities(search_query)
                entities = [e for e in entities if e['type'] in entity_types]
            else:
                entities = st.session_state.detector.knowledge_graph.get_entities_by_type(entity_types)
            
            if entities:
                for entity in entities:
                    with st.expander(f"{entity['type']}: {entity['name']}"):
                        # Display entity details
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Relationships:**")
                            st.json(entity['relationships'])
                        with col2:
                            if 'mentions' in entity['relationships']:
                                st.write("**Mention Count:**", len(entity['relationships']['mentions']))
                            if 'sources' in entity['relationships']:
                                st.write("**Source Count:**", len(entity['relationships']['sources']))
            else:
                st.info("No entities found matching the criteria.")
        else:
            st.warning("Please select at least one entity type.")
        
        # Add export selected entities option
        if 'entities' in locals():  # Check if entities exists
            if entities:  # Check if entities is not empty
                if st.button("Export Selected Entities"):
                    export_data = {
                        "entities": entities,
                        "export_time": datetime.now().isoformat(),
                        "filters": {
                            "types": entity_types,
                            "search_query": search_query if search_query else "None"
                        }
                    }
                    
                    # Convert to JSON and offer download
                    json_str = json.dumps(export_data, indent=2)
                    st.download_button(
                        label="Download Entities JSON",
                        data=json_str,
                        file_name=f"entities_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                          
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
            "🕸️ Knowledge Graph",
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
    
    elif nav_selection == "🕸️ Knowledge Graph":
        knowledge_graph_tab()
        
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