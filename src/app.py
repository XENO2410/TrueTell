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
import networkx as nx
from broadcast.stream import BroadcastStream, BroadcastMessage
from broadcast.analyzer import BroadcastAnalyzer

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

async def start_monitoring(source_type, settings, monitoring_speed):
    """Start monitoring based on source type"""
    try:
        while True:
            items = []
            
            if source_type in ["All Sources", "News API"]:
                news_items = await st.session_state.broadcast_stream.news_source.fetch_newsapi()
                items.extend(news_items)
                
            if source_type in ["All Sources", "Guardian"]:
                guardian_items = await st.session_state.broadcast_stream.news_source.fetch_guardian()
                items.extend(guardian_items)
                
            if source_type in ["All Sources", "Twitter"]:
                social_items = await st.session_state.broadcast_stream.social_monitor.monitor_twitter()
                items.extend(social_items)
            
            # Process items
            for item in items:
                message = BroadcastMessage(
                    text=item['text'],
                    timestamp=item['timestamp'],
                    source=item['source'],
                    metadata=item['metadata']
                )
                await st.session_state.broadcast_analyzer.process_message(message)
            
            # Wait before next fetch
            await asyncio.sleep(monitoring_speed)
            
    except Exception as e:
        st.error(f"Error during monitoring: {str(e)}")
        
def live_monitor_tab():
    st.title("📡 Live Broadcast Monitor")
    
    if 'monitoring_active' not in st.session_state:
        st.session_state.monitoring_active = False
    
    if 'monitoring_results' not in st.session_state:
        st.session_state.monitoring_results = []

    # Monitoring controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        monitoring_speed = st.slider("Update Interval (seconds)", 5, 30, 10)
    
    with col2:
        if not st.session_state.monitoring_active:
            if st.button("▶️ Start Monitoring"):
                st.session_state.monitoring_active = True
                st.rerun()
        else:
            if st.button("⏹️ Stop Monitoring"):
                st.session_state.monitoring_active = False
                st.rerun()
    
    with col3:
        if st.button("🔄 Reset"):
            st.session_state.monitoring_active = False
            st.session_state.monitoring_results = []
            st.rerun()

    # Create placeholders
    status_placeholder = st.empty()
    news_container = st.container()

    # Show export button for flagged articles
    if st.session_state.monitoring_results:
        try:
            export_data = []
            for r in st.session_state.monitoring_results:
                # Extract data with fallbacks for missing keys
                export_item = {
                    'timestamp': r.get('timestamp', ''),
                    'title': r.get('title', ''),
                    'source': r.get('source', ''),
                    'risk_score': 0.0,
                    'credibility_score': 0.0
                }
                
                # Get text from either 'text' key or 'sentence' key
                export_item['text'] = r.get('text', r.get('sentence', ''))
                
                # Get analysis scores if available
                analysis = r.get('analysis', {})
                if analysis:
                    export_item['risk_score'] = analysis.get('risk_score', 0) * 100
                    fact_check = analysis.get('fact_check_results', {})
                    export_item['credibility_score'] = fact_check.get('credibility_score', 0) * 100
                
                export_data.append(export_item)
            
            # Create DataFrame and export
            st.download_button(
                "📥 Export Results",
                data=pd.DataFrame(export_data).to_csv(index=False),
                file_name=f"news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv'
            )
        except Exception as e:
            st.error(f"Error preparing export: {str(e)}")

    # Monitoring logic
    if st.session_state.monitoring_active:
        status_placeholder.info("🔄 Monitoring active...")
        
        try:
            async def monitoring_loop():
                while st.session_state.monitoring_active:
                    articles = await st.session_state.broadcast_stream.news_source.fetch_news()
                    
                    if articles:
                        with news_container:
                            for article in articles:
                                # Create message
                                message = BroadcastMessage(
                                    text=article['text'],
                                    timestamp=article['timestamp'],
                                    source=article['source'],
                                    metadata={
                                        'url': article.get('url', ''),
                                        'author': article.get('author', '')
                                    }
                                )
                                
                                # Analyze content
                                analysis = await st.session_state.broadcast_analyzer.process_message(message)
                                
                                if analysis:
                                    # Store result
                                    result = {**article, 'analysis': analysis}
                                    st.session_state.monitoring_results.append(result)
                                    
                                    # Display article in a card-like format
                                    st.markdown("---")  # Separator between articles
                                    
                                    # Article header
                                    st.markdown(f"### 📰 {article['title']}")
                                    
                                    # Article content and analysis
                                    col1, col2 = st.columns([3, 1])
                                    
                                    with col1:
                                        st.markdown(f"**Source:** {article['source']}")
                                        st.markdown(article['text'])
                                        st.markdown(f"*Published: {article['timestamp']}*")
                                        if article.get('url'):
                                            st.markdown(f"[Read more]({article['url']})")
                                    
                                    with col2:
                                        risk_score = analysis.get('risk_score', 0) * 100
                                        credibility_score = analysis.get('fact_check_results', {}).get('credibility_score', 0) * 100
                                        
                                        # Risk level indicator
                                        if risk_score > 70:
                                            st.error(f"High Risk: {risk_score:.1f}%")
                                        elif risk_score > 40:
                                            st.warning(f"Medium Risk: {risk_score:.1f}%")
                                        else:
                                            st.success(f"Low Risk: {risk_score:.1f}%")
                                        
                                        st.metric("Credibility", f"{credibility_score:.1f}%")
                                        
                                        if risk_score > 70:
                                            st.warning("⚠️ Potential misinformation detected!")
                                    
                                    # Analysis details in tabs instead of expander
                                    tab1, tab2 = st.tabs(["Key Points", "Full Analysis"])
                                    
                                    with tab1:
                                        st.markdown("**Key Findings:**")
                                        st.markdown(f"- Risk Level: {'High' if risk_score > 70 else 'Medium' if risk_score > 40 else 'Low'}")
                                        st.markdown(f"- Credibility Score: {credibility_score:.1f}%")
                                        if analysis.get('key_terms'):
                                            st.markdown(f"- Key Terms: {', '.join(analysis['key_terms'])}")
                                    
                                    with tab2:
                                        st.json(analysis)
                    
                    await asyncio.sleep(monitoring_speed)
            
            # Run the monitoring loop
            asyncio.run(monitoring_loop())
            
        except Exception as e:
            st.error(f"Monitoring error: {str(e)}")
            st.session_state.monitoring_active = False

def display_monitoring_stats(results):
    """Display monitoring statistics"""
    if not results:
        return
        
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Items", len(results))
    with col2:
        high_risk = sum(1 for r in results if r['analysis'].get('risk_score', 0) > 0.7)
        st.metric("High Risk Items", high_risk)
    with col3:
        avg_risk = np.mean([r['analysis'].get('risk_score', 0) for r in results]) * 100
        st.metric("Average Risk", f"{avg_risk:.1f}%")
    with col4:
        avg_cred = np.mean([r['analysis'].get('credibility_score', 0) for r in results]) * 100
        st.metric("Average Credibility", f"{avg_cred:.1f}%")

def export_results_as_csv(results):
    """Convert monitoring results to CSV"""
    df = pd.DataFrame([{
        'timestamp': r.get('timestamp', ''),
        'source': r.get('source', ''),
        'text': r.get('text', ''),
        'risk_score': r.get('analysis', {}).get('risk_score', 0) * 100,
        'credibility_score': r.get('analysis', {}).get('credibility_score', 0) * 100,
        'alerts': len([a for a in st.session_state.detector.alert_system.alerts 
                      if a.source_text == r.get('text', '')])
    } for r in results])
    
    return df.to_csv(index=False)

async def process_broadcast_item(item, placeholder):
    """Process a single broadcast item"""
    try:
        # Create broadcast message
        message = BroadcastMessage(
            text=item['text'],
            timestamp=item['timestamp'],
            source=item['source'],
            metadata=item['metadata']
        )

        # Analyze the message
        analysis_result = await st.session_state.broadcast_analyzer.process_message(message)

        # Add to monitoring results
        if analysis_result:
            st.session_state.monitoring_results.append({
                **item,
                'analysis': analysis_result
            })

        # Update display
        with placeholder.container():
            # Show latest message
            st.subheader("Latest Message")
            st.info(f"[{datetime.now().strftime('%H:%M:%S')}] {message.text}")

            # Show analysis results
            if analysis_result:
                # Alert status
                if analysis_result.get('risk_score', 0) > 0.7:
                    st.error("🚨 High-risk content detected!")

                # Display results
                display_live_results(st.session_state.monitoring_results)

    except Exception as e:
        st.error(f"Error processing broadcast item: {str(e)}")
        
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

def handle_export_import():
    """Handle export and import functionality"""
    col1, col2 = st.columns(2)
    
    # Export functionality
    with col1:
        export_format = st.selectbox("Export Format", ["JSON", "PNG", "HTML"])
        if st.button("Export Graph"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                filename = f"knowledge_graph_{timestamp}.{export_format.lower()}"
                if export_format == "JSON":
                    st.session_state.detector.knowledge_graph.export_graph(filename)
                    mime_type = "application/json"
                    mode = 'r'
                elif export_format == "PNG":
                    st.session_state.detector.knowledge_graph.export_as_image(filename)
                    mime_type = "image/png"
                    mode = 'rb'
                else:  # HTML
                    st.session_state.detector.knowledge_graph.export_as_html(filename)
                    mime_type = "text/html"
                    mode = 'r'
                
                with open(filename, mode) as f:
                    st.download_button(
                        label=f"Download {export_format}",
                        data=f.read(),
                        file_name=filename,
                        mime=mime_type
                    )
                st.success(f"✅ Graph exported as {export_format}")
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
    
    # Import functionality
    with col2:
        uploaded_file = st.file_uploader("Import Graph", type="json")
        if uploaded_file is not None:
            try:
                with open("temp_graph.json", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.detector.knowledge_graph.import_graph("temp_graph.json")
                st.success("✅ Graph imported successfully!")
            except Exception as e:
                st.error(f"❌ Import failed: {str(e)}")

def handle_visualization_settings():
    """Handle visualization settings"""
    with st.expander("Visualization Settings"):
        col_a, col_b = st.columns(2)
        with col_a:
            min_weight = st.slider("Minimum Relationship Weight", 0.0, 1.0, 0.1)
            show_labels = st.checkbox("Show Node Labels", value=True)
        with col_b:
            layout_type = st.selectbox("Layout Type", ["spring", "circular", "random"])
            node_size = st.slider("Node Size", 10, 50, 20)
    return min_weight, show_labels, layout_type, node_size

def display_graph_statistics(stats):
    """Display graph statistics"""
    col_stats1, col_stats2 = st.columns(2)
    with col_stats1:
        st.metric("🔵 Nodes", stats['total_nodes'])
        st.metric("🔗 Edges", stats['total_edges'])
    with col_stats2:
        st.metric("📊 Density", round(stats.get('graph_density', 0), 3))
        st.metric("🔄 Clusters", stats.get('num_clusters', 0))
    
    if 'node_types' in stats:
        st.write("**Node Distribution:**")
        max_count = max(stats['node_types'].values())
        for node_type, count in stats['node_types'].items():
            st.write(f"{node_type}")
            st.progress(count / max_count)
            st.caption(f"Count: {count}")
    
    with st.expander("Advanced Metrics"):
        st.write(f"**Avg. Degree:** {stats.get('avg_degree', 0):.2f}")
        st.write(f"**Diameter:** {stats.get('diameter', 0)}")
        st.write(f"**Avg. Path Length:** {stats.get('avg_path_length', 0):.2f}")

def handle_pattern_analysis(analysis_type):
    """Handle pattern analysis based on type"""
    params = {}
    if analysis_type == "Entity Co-occurrence":
        params['min_occurrence'] = st.slider("Minimum Occurrence", 1, 10, 2)
        params['entity_types'] = st.multiselect(
            "Entity Types to Analyze",
            options=list(st.session_state.detector.knowledge_graph.entity_types),
            default=["PERSON", "ORG"]
        )
    elif analysis_type == "Temporal Patterns":
        params['time_window'] = st.selectbox("Time Window", ["Hour", "Day", "Week", "Month"])
        params['include_weekends'] = st.checkbox("Include Weekends", value=True)
    else:  # Narrative Chains
        params['chain_length'] = st.slider("Minimum Chain Length", 2, 10, 3)
        params['similarity_threshold'] = st.slider("Similarity Threshold", 0.0, 1.0, 0.7)
    return params

def knowledge_graph_tab():
    st.title("🕸️ Knowledge Graph Analysis")
    
    # Add sidebar controls for Export/Import
    st.sidebar.markdown("### Knowledge Graph Controls")
    
    # Export/Import section
    with st.sidebar.expander("Export/Import"):
        col1, col2 = st.columns(2)
        
        # Export functionality
        with col1:
            export_format = st.selectbox("Export Format", ["JSON", "PNG", "HTML"])
            if st.button("Export Graph"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                try:
                    if export_format == "JSON":
                        filename = f"knowledge_graph_{timestamp}.json"
                        st.session_state.detector.knowledge_graph.export_graph(filename)
                        with open(filename, 'r') as f:
                            st.download_button(
                                label="Download JSON",
                                data=f.read(),
                                file_name=filename,
                                mime="application/json"
                            )
                    elif export_format == "PNG":
                        filename = f"knowledge_graph_{timestamp}.png"
                        st.session_state.detector.knowledge_graph.export_as_image(filename)
                        with open(filename, 'rb') as f:
                            st.download_button(
                                label="Download PNG",
                                data=f.read(),
                                file_name=filename,
                                mime="image/png"
                            )
                    else:  # HTML
                        filename = f"knowledge_graph_{timestamp}.html"
                        st.session_state.detector.knowledge_graph.export_as_html(filename)
                        with open(filename, 'r') as f:
                            st.download_button(
                                label="Download HTML",
                                data=f.read(),
                                file_name=filename,
                                mime="text/html"
                            )
                    st.success(f"✅ Graph exported as {export_format}")
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")
        
        # Import functionality
        with col2:
            uploaded_file = st.file_uploader("Import Graph", type="json")
            if uploaded_file is not None:
                try:
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
            
            # Add visualization controls
            with st.expander("Visualization Settings"):
                col_a, col_b = st.columns(2)
                with col_a:
                    min_weight = st.slider("Minimum Relationship Weight", 0.0, 1.0, 0.1)
                    show_labels = st.checkbox("Show Node Labels", value=True)
                with col_b:
                    layout_type = st.selectbox("Layout Type", ["spring", "circular", "random"])
                    node_size = st.slider("Node Size", 10, 50, 20)
            
            # Add search and highlight
            search_term = st.text_input("🔍 Search and Highlight Nodes", "")
            highlighted_nodes = []
            if search_term:
                highlighted_nodes = st.session_state.detector.knowledge_graph.search_nodes(search_term)
            
            # Visualize graph with settings
            st.session_state.detector.knowledge_graph.visualize(
                min_weight=min_weight,
                show_labels=show_labels,
                layout=layout_type,
                node_size=node_size,
                highlight_nodes=highlighted_nodes
            )
            
            # Add refresh button
            if st.button("🔄 Refresh Visualization"):
                st.rerun()
        
        with col2:
            st.subheader("Graph Statistics")
            stats = st.session_state.detector.knowledge_graph.get_statistics()
            
            # Display metrics with enhanced styling
            col_stats1, col_stats2 = st.columns(2)
            with col_stats1:
                st.metric("🔵 Nodes", stats['total_nodes'])
                st.metric("🔗 Edges", stats['total_edges'])
            with col_stats2:
                st.metric("📊 Density", round(stats.get('graph_density', 0), 3))
                st.metric("🔄 Clusters", stats.get('num_clusters', 0))
            
            # Display node type distribution with progress bars
            if 'node_types' in stats:
                st.write("**Node Distribution:**")
                max_count = max(stats['node_types'].values())
                for node_type, count in stats['node_types'].items():
                    st.write(f"{node_type}")
                    st.progress(count / max_count)
                    st.caption(f"Count: {count}")
            
            # Add graph metrics
            with st.expander("Advanced Metrics"):
                st.write(f"**Avg. Degree:** {stats.get('avg_degree', 0):.2f}")
                st.write(f"**Diameter:** {stats.get('diameter', 0)}")
                st.write(f"**Avg. Path Length:** {stats.get('avg_path_length', 0):.2f}")
    
    with tab2:
        st.subheader("Pattern Analysis")
        
        # Add analysis options with enhanced UI
        analysis_type = st.radio(
            "Select Analysis Type",
            ["Entity Co-occurrence", "Temporal Patterns", "Narrative Chains"],
            horizontal=True
        )
        
        # Add analysis parameters
        params = {}  # Initialize params dictionary
        with st.expander("Analysis Parameters"):
            if analysis_type == "Entity Co-occurrence":
                params['min_occurrence'] = st.slider("Minimum Occurrence", 1, 10, 2)
                params['entity_types'] = st.multiselect(
                    "Entity Types to Analyze",
                    options=list(st.session_state.detector.knowledge_graph.entity_types),
                    default=["PERSON", "ORG"]
                )
            elif analysis_type == "Temporal Patterns":
                params['time_window'] = st.selectbox("Time Window", ["Hour", "Day", "Week", "Month"])
                params['include_weekends'] = st.checkbox("Include Weekends", value=True)
            else:  # Narrative Chains
                params['chain_length'] = st.slider("Minimum Chain Length", 2, 10, 2)
                params['similarity_threshold'] = st.slider("Similarity Threshold", 0.0, 1.0, 0.3)
        
        if st.button("Analyze Patterns"):
            with st.spinner("Analyzing patterns..."):
                try:
                    # Get patterns with parameters
                    patterns = st.session_state.detector.knowledge_graph.analyze_claim_patterns(
                        analysis_type=analysis_type,
                        **params
                    )
                    
                    if patterns is None:
                        st.error("Error: Pattern analysis returned no results")
                        return
                    
                    # Display metadata first
                    if patterns.get('metadata'):  # Changed from analysis_metadata
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Claims", patterns['metadata']['total_claims'])
                        with col2:
                            st.metric("Average Credibility", f"{patterns['metadata']['avg_credibility']:.2%}")
                        with col3:
                            st.metric("Risk Level", patterns['metadata']['risk_level'])
                    
                    # Display results based on analysis type
                    if analysis_type == "Entity Co-occurrence":
                        st.write("**Entity Co-occurrence Analysis**")
                        if patterns.get('common_entities'):
                            sorted_entities = dict(sorted(
                                patterns['common_entities'].items(),
                                key=lambda x: x[1],
                                reverse=True
                            ))
                            
                            # Create bar chart for common entities
                            fig = px.bar(
                                x=list(sorted_entities.keys())[:10],
                                y=list(sorted_entities.values())[:10],
                                title="Most Common Entities",
                                labels={'x': 'Entity', 'y': 'Frequency'},
                                template="plotly_dark"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display co-occurrence network
                            if patterns.get('co_occurrence_network'):
                                st.write("**Entity Co-occurrence Network**")
                                st.session_state.detector.knowledge_graph.visualize_network(
                                    patterns['co_occurrence_network']
                                )
                                
                                # Add network statistics
                                network = patterns['co_occurrence_network']
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Connections", network.number_of_edges())
                                with col2:
                                    st.metric("Unique Entities", network.number_of_nodes())
                                with col3:
                                    density = nx.density(network)
                                    st.metric("Network Density", f"{density:.3f}")
                        else:
                            st.info("No entity co-occurrence data available.")
                    
                    elif analysis_type == "Temporal Patterns":
                        st.write("**Temporal Distribution Analysis**")
                        if patterns.get('temporal_patterns'):
                            # Display time range with error handling
                            try:
                                start_time = patterns['temporal_metadata'].get('start_time', 'N/A')
                                end_time = patterns['temporal_metadata'].get('end_time', 'N/A')
                                st.info(f"Analysis Period: {start_time} to {end_time}")
                            except Exception as e:
                                st.warning("Time range information not available")
                            
                            # Convert temporal patterns to DataFrame
                            temporal_data = []
                            for timestamp_str, count in patterns['temporal_patterns'].items():
                                try:
                                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                                    temporal_data.append({'timestamp': timestamp, 'count': count})
                                except Exception as e:
                                    st.warning(f"Error parsing timestamp {timestamp_str}: {e}")
                                    continue
                            
                            if temporal_data:
                                temporal_df = pd.DataFrame(temporal_data)
                                temporal_df = temporal_df.sort_values('timestamp')
                                
                                # Timeline visualization
                                fig = px.line(
                                    temporal_df,
                                    x='timestamp',
                                    y='count',
                                    title=f"Content Distribution Over {params.get('time_window', 'Time')}",
                                    labels={'timestamp': 'Time', 'count': 'Number of Claims'},
                                    template="plotly_dark"
                                )
                                
                                # Enhance the visualization
                                fig.update_traces(
                                    line_color='#00ff00',
                                    mode='lines+markers',  # Add markers
                                    marker=dict(
                                        size=10,
                                        symbol='circle'
                                    )
                                )
                                
                                # Customize x-axis
                                fig.update_xaxes(
                                    dtick="H1",  # Show every hour
                                    tickformat="%H:%M",  # Show hour:minute format
                                    gridcolor='rgba(128, 128, 128, 0.2)',  # Lighter grid
                                    title_text="Time"
                                )
                                
                                # Customize y-axis
                                fig.update_yaxes(
                                    gridcolor='rgba(128, 128, 128, 0.2)',
                                    title_text="Number of Claims"
                                )
                                
                                # Update layout
                                fig.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='white'),
                                    showlegend=False,
                                    hovermode='x unified'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Display metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Events", patterns['temporal_metadata']['total_entries'])
                                with col2:
                                    st.metric("Peak Count", patterns['temporal_metadata']['max_value'])
                                with col3:
                                    st.metric("Average Count", f"{patterns['temporal_metadata']['average_per_slot']:.1f}")
                                
                                # Display heatmap
                                if patterns.get('temporal_heatmap'):
                                    st.write("**Activity Heatmap**")
                                    heatmap_data = np.array(patterns['temporal_heatmap'])
                                    
                                    # Get days and hours from metadata
                                    days = patterns['temporal_metadata'].get('days', ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
                                    hours = patterns['temporal_metadata'].get('hours', [f"{h:02d}:00" for h in range(24)])
                                    
                                    fig_heatmap = px.imshow(
                                        heatmap_data,
                                        labels=dict(x="Hour of Day", y="Day of Week"),
                                        x=hours,
                                        y=days,
                                        title="Activity Distribution (Claims per Hour)",
                                        color_continuous_scale="Viridis"
                                    )
                                    st.plotly_chart(fig_heatmap, use_container_width=True)
                                    
                                    # Display distribution statistics
                                    if 'distribution' in patterns['temporal_metadata']:
                                        with st.expander("Distribution Statistics"):
                                            st.json(patterns['temporal_metadata']['distribution'])
                            else:
                                st.warning("No valid temporal data to display")
                        else:
                            st.info("No temporal pattern data available.")
                    
                    elif analysis_type == "Narrative Chains":
                        st.write("**Narrative Chain Analysis**")
                        if patterns.get('narrative_chains'):
                            for idx, chain in enumerate(patterns['narrative_chains']):
                                st.subheader(f"Narrative Chain {idx+1} ({chain['length']} claims)")
                                
                                # Display chain metrics in columns instead of expander
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Chain Length", chain['length'])
                                with col2:
                                    st.metric("Average Credibility", f"{chain['avg_credibility']:.2%}")
                                with col3:
                                    st.metric("Claims", len(chain['claims']))
                                
                                # Display claims in a regular container
                                st.write("**Claims in Chain:**")
                                for claim in chain['claims']:
                                    container = st.container()
                                    with container:
                                        col1, col2 = st.columns([3, 1])
                                        with col1:
                                            st.write(claim['text'])
                                        with col2:
                                            st.progress(claim['credibility_score'])
                                            st.caption(f"Credibility: {claim['credibility_score']:.2%}")
                                
                                # Visualize chain
                                if chain.get('chain_graph'):
                                    st.write("**Chain Visualization**")
                                    st.session_state.detector.knowledge_graph.visualize_chain(
                                        chain['chain_graph']
                                    )
                                
                                # Display statistics in a regular container
                                st.write("**Chain Statistics**")
                                stats_df = pd.DataFrame([{
                                    'Claim ID': c['claim_id'],
                                    'Timestamp': c['timestamp'],
                                    'Credibility': c['credibility_score'],
                                    'Entities': len(c['entities'])
                                } for c in chain['claims']])
                                st.dataframe(stats_df)
                                
                                # Add a separator between chains
                                if idx < len(patterns['narrative_chains']) - 1:
                                    st.markdown("---")
                        else:
                            st.info("No narrative chains found.")
                            
                except Exception as e:
                    st.error(f"Error during pattern analysis: {str(e)}")
                    st.exception(e)  # This will show the full traceback
    
    with tab3:
        st.subheader("Entity Explorer")
        
        # Enhanced search functionality
        col_search1, col_search2 = st.columns([2, 1])
        with col_search1:
            search_query = st.text_input("🔍 Search Entities", "")
        with col_search2:
            search_type = st.selectbox("Search Type", ["Contains", "Exact Match", "Regex"])
        
        # Enhanced entity type filter
        entity_types = st.multiselect(
            "Filter by Entity Type",
            options=list(st.session_state.detector.knowledge_graph.entity_types),
            default=["PERSON", "ORG"]
        )
        
        # Add relationship type filter
        relationship_types = st.multiselect(
            "Filter by Relationship Type",
            options=list(st.session_state.detector.knowledge_graph.relationship_types),
            default=[]
        )
        
        # Display entities with enhanced UI
        if entity_types:
            entities = st.session_state.detector.knowledge_graph.search_entities(
                query=search_query,
                search_type=search_type,
                entity_types=entity_types,
                relationship_types=relationship_types
            )
            
            if entities:
                # Add sorting options
                sort_by = st.selectbox(
                    "Sort By",
                    ["Name", "Type", "Relationship Count", "Mention Count"]
                )
                
                # Sort entities based on selection
                entities = st.session_state.detector.knowledge_graph.sort_entities(
                    entities,
                    sort_by=sort_by
                )
                
                # Display entities with enhanced UI
                for entity in entities:
                    with st.expander(f"{entity['type']}: {entity['name']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Relationships:**")
                            st.json(entity['relationships'])
                            
                            # Add relationship visualization
                            if st.button(f"Visualize Relationships for {entity['name']}"):
                                st.session_state.detector.knowledge_graph.visualize_entity_relationships(
                                    entity['name']
                                )
                        
                        with col2:
                            # Enhanced metrics
                            st.write("**Entity Metrics:**")
                            if 'mentions' in entity['relationships']:
                                st.metric(
                                    "Mentions",
                                    len(entity['relationships']['mentions']),
                                    delta=entity.get('mention_trend', 0)
                                )
                            if 'sources' in entity['relationships']:
                                st.metric(
                                    "Sources",
                                    len(entity['relationships']['sources']),
                                    delta=entity.get('source_trend', 0)
                                )
                            
                            # Add temporal distribution
                            if 'temporal_distribution' in entity:
                                st.write("**Temporal Distribution:**")
                                st.line_chart(entity['temporal_distribution'])
                
                # Add export selected entities option
                if st.button("Export Selected Entities"):
                    export_data = {
                        "entities": entities,
                        "export_time": datetime.now().isoformat(),
                        "filters": {
                            "types": entity_types,
                            "search_query": search_query if search_query else "None",
                            "search_type": search_type,
                            "relationship_types": relationship_types
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
            else:
                st.info("No entities found matching the criteria.")
        else:
            st.warning("Please select at least one entity type.")
                          
def main():
    # Must be the first Streamlit command
    st.set_page_config(page_title="Real-time Misinformation Detector", layout="wide")
    
    # Then add custom CSS
    st.markdown("""
        <style>
        .sidebar-nav {
            padding: 10px;
        }
        .nav-link {
            padding: 8px 15px;
            margin: 5px 0;
            border-radius: 5px;
            background-color: rgba(255, 255, 255, 0.1);
            color: #fff;
            text-decoration: none;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .nav-link:hover {
            background-color: rgba(255, 255, 255, 0.2);
            transform: translateX(5px);
        }
        .nav-link.active {
            background-color: rgba(255, 255, 255, 0.2);
            border-left: 4px solid #ff4b4b;
        }
        .system-status {
            padding: 15px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            margin: 10px 0;
        }
        .about-section {
            padding: 15px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            margin: 10px 0;
        }
        .help-section {
            margin-top: 20px;
        }
        .version-tag {
            font-size: 0.8em;
            opacity: 0.7;
            text-align: center;
            margin-top: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state objects
    if 'detector' not in st.session_state:
        st.session_state.detector = MisinformationDetector()
    
    if 'dashboard_manager' not in st.session_state:
        st.session_state.dashboard_manager = DashboardManager()
        
    if 'broadcast_stream' not in st.session_state:
        news_api_key = os.getenv('NEWS_API_KEY')
        if not news_api_key:
            st.error("NEWS_API_KEY not found in environment variables")
            st.stop()
        
        st.session_state.broadcast_stream = BroadcastStream()
        
    if 'broadcast_analyzer' not in st.session_state:
        st.session_state.broadcast_analyzer = BroadcastAnalyzer(
            st.session_state.detector,
            st.session_state.detector.knowledge_graph
        )
    
    # Sidebar Header with Logo
    st.sidebar.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="color: #ff4b4b;">TruthTell</h1>
        </div>
    """, unsafe_allow_html=True)

    # Navigation Options
    nav_options = {
        "🔍 Text Analysis": "text_analysis",
        "📡 Live Monitor": "live_monitor",
        "📊 Dashboard": "dashboard",
        "🚨 Alerts": "alerts",
        "🕸️ Knowledge Graph": "knowledge_graph",
        "🔌 Integrations": "integrations"
    }

    # Store the current page in session state if not present
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "text_analysis"

    # Create navigation menu with custom styling
    for label, page in nav_options.items():
        button_style = """
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
            padding: 10px;
            margin: 5px 0;
            transition: all 0.3s ease;
        """
        if st.session_state.current_page == page:
            button_style += """
                border-left: 4px solid #ff4b4b;
                background-color: rgba(255, 255, 255, 0.2);
            """
        
        if st.sidebar.button(
            label,
            key=f"nav_{page}",
            use_container_width=True,
            help=f"Navigate to {label}"
        ):
            st.session_state.current_page = page
            st.rerun()

    # System Status with Enhanced UI
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        <div class="system-status">
            <h3>System Status</h3>
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 10px; height: 10px; background: #00ff00; border-radius: 50%;"></div>
                <span>System Online</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.metric(
        "Active Alerts",
        len(st.session_state.detector.alert_system.alerts),
        delta="real-time"
    )
    
    # Navigation logic based on session state
    if st.session_state.current_page == "text_analysis":
        text_analysis_tab()
    elif st.session_state.current_page == "live_monitor":
        live_monitor_tab()
    elif st.session_state.current_page == "dashboard":
        dashboard_tab()
    elif st.session_state.current_page == "alerts":
        alerts_tab()
    elif st.session_state.current_page == "knowledge_graph":
        knowledge_graph_tab()
    elif st.session_state.current_page == "integrations":
        integration_settings_tab()
    
    # Enhanced About Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        <div class="about-section">
            <h3>About</h3>
            <p> This is a real-time misinformation detection system.</p>
            <div class="version-tag">Version 1.0.0</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Help Section
    with st.sidebar.expander("📚 Quick Guide"):
        st.markdown("""
            <div class="help-section">
                <ol style="padding-left: 20px;">
                    <li>🔍 Use <b>Text Analysis</b> for single text analysis</li>
                    <li>📡 Use <b>Live Monitor</b> for real-time monitoring</li>
                    <li>📊 Check <b>Dashboard</b> for overall statistics</li>
                    <li>🚨 Monitor <b>Alerts</b> for important notifications</li>
                    <li>🕸️ View <b>Knowledge Graph</b> for claim relationships</li>
                    <li>🔌 Configure <b>Integrations</b> for external connections</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
        
if __name__ == "__main__":
    main()