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
from utils import download_nltk_data
from fact_checker import FactChecker
from source_checker import SourceChecker
from realtime_processor import RealTimeProcessor
from dotenv import load_dotenv

# Load environment variables and download NLTK data
load_dotenv()
download_nltk_data()

class MisinformationDetector:
    def __init__(self):
        self.classifier = pipeline("zero-shot-classification", 
                                 model="facebook/bart-large-mnli")
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.stop_words = set(stopwords.words('english'))
        self.fact_checker = FactChecker()
        self.source_checker = SourceChecker()
        
    def preprocess_text(self, text: str) -> List[str]:
        return sent_tokenize(text)
    
    def extract_key_terms(self, text: str) -> List[str]:
        try:
            tokens = word_tokenize(text)
            
            basic_terms = [word for word in tokens 
                          if word.isalnum() 
                          and word.lower() not in self.stop_words
                          and len(word) > 2]
            
            try:
                pos_tags = nltk.pos_tag(tokens)
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
            
            results.append({
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
            })
            
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

def display_live_results(results: List[Dict]):
    if not results:
        st.warning("No results to display yet. Start monitoring to see analysis.")
        return

    col1, col2, col3 = st.columns(3)
    
    avg_risk = np.mean([r['risk_score'] for r in results])
    avg_fact_check = np.mean([r['fact_check_results']['credibility_score'] for r in results])
    
    with col1:
        st.metric("Average Risk Score", f"{avg_risk:.2%}")
    
    with col2:
        st.metric("Average Fact-Check Score", f"{avg_fact_check:.2%}")
    
    with col3:
        high_risk_count = sum(1 for r in results if r['risk_score'] > 0.7)
        st.metric("High Risk Statements", high_risk_count)
    
    # Create timeline chart
    if len(results) > 0:
        df = pd.DataFrame([{
            'timestamp': datetime.strptime(r['timestamp'], "%Y-%m-%d %H:%M:%S"),
            'risk_score': r['risk_score'],
            'text': r['sentence'][:50] + '...'
        } for r in results])
        
        fig_timeline = px.line(
            df,
            x='timestamp',
            y='risk_score',
            title='Risk Score Timeline',
            hover_data=['text']
        )
        st.plotly_chart(fig_timeline)
    
    # Display latest analyses
    st.subheader("Latest Analyses")
    for result in results[-5:]:  # Show last 5 results
        with st.expander(f"Analysis for: {result['sentence'][:100]}..."):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Classification Breakdown:**")
                for cls, score in zip(result['classifications'], result['classification_scores']):
                    st.write(f"- {cls}: {score:.2%}")
                st.write("**Key Terms:**", ", ".join(result['key_terms']))
            
            with col2:
                st.metric("Risk Score", f"{result['risk_score']:.2%}")
                if result['risk_score'] > 0.7:
                    st.error("⚠️ High Risk Content!")
                elif result['risk_score'] > 0.4:
                    st.warning("⚠️ Medium Risk Content")

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
    
    source_type = st.selectbox(
        "Select Source",
        ["Live News Feed", "Social Media Stream", "Custom Input"]
    )
    
    if 'monitoring_results' not in st.session_state:
        st.session_state.monitoring_results = []
    
    placeholder = st.empty()
    
    if source_type == "Live News Feed":
        if st.button("Start Monitoring"):
            # Sample live feed data
            live_feeds = [
                "Breaking news: Major technological breakthrough announced in renewable energy.",
                "Scientists discover new species in the Amazon rainforest.",
                "Controversial statement: Social media platform implements new policy changes.",
                "ALERT: Unverified reports of unusual weather patterns emerging globally.",
                "Latest update: Stock market shows unexpected movements after policy announcement."
            ]
            
            with st.spinner("Monitoring live feed..."):
                for feed in live_feeds:
                    # Analyze the feed using existing detector
                    results = st.session_state.detector.analyze_text(feed)
                    st.session_state.monitoring_results.extend(results)
                    
                    # Update display
                    with placeholder.container():
                        display_live_results(st.session_state.monitoring_results)
                        
                        # Show latest feed
                        st.subheader("Latest Feed")
                        st.info(feed)
                        
                        # Show real-time metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Active Feeds", len(st.session_state.monitoring_results))
                        with col2:
                            avg_risk = np.mean([r['risk_score'] for r in st.session_state.monitoring_results])
                            st.metric("Average Risk", f"{avg_risk:.2%}")
                        with col3:
                            high_risk = sum(1 for r in st.session_state.monitoring_results if r['risk_score'] > 0.7)
                            st.metric("High Risk Alerts", high_risk)
                    
                    time.sleep(2)  # Simulate real-time delay
    
    elif source_type == "Social Media Stream":
        if st.button("Start Monitoring"):
            # Sample social media posts
            social_posts = [
                "Just heard that AI will replace all jobs by next year! #tech #future",
                "New study shows regular exercise improves mental health. #health #wellness",
                "SHOCKING: Celebrity reveals conspiracy theory about government! #news",
                "Beautiful sunset at the beach today! 🌅 #nature #peace",
                "This miracle product cures everything! Buy now! #health #promotion"
            ]
            
            with st.spinner("Monitoring social media..."):
                for post in social_posts:
                    results = st.session_state.detector.analyze_text(post)
                    st.session_state.monitoring_results.extend(results)
                    
                    with placeholder.container():
                        display_live_results(st.session_state.monitoring_results)
                        
                        st.subheader("Latest Social Media Post")
                        st.info(post)
                        
                        # Show hashtag analysis
                        hashtags = [tag for tag in post.split() if tag.startswith('#')]
                        if hashtags:
                            st.write("Trending Hashtags:", ', '.join(hashtags))
                    
                    time.sleep(2)
    
    elif source_type == "Custom Input":
        custom_input = st.text_area("Enter custom text to monitor:")
        if st.button("Analyze Custom Input"):
            if custom_input:
                results = st.session_state.detector.analyze_text(custom_input)
                st.session_state.monitoring_results.extend(results)
                
                with placeholder.container():
                    display_live_results(st.session_state.monitoring_results)
    
    # Add reset button
    if st.button("Reset Monitoring"):
        st.session_state.monitoring_results = []
        placeholder.empty()

def main():
    st.set_page_config(page_title="Real-time Misinformation Detector", 
                       layout="wide")
    
    if 'detector' not in st.session_state:
        st.session_state.detector = MisinformationDetector()
    
    tab1, tab2 = st.tabs(["Text Analysis", "Live Monitor"])
    
    with tab1:
        text_analysis_tab()
    
    with tab2:
        live_monitor_tab()

if __name__ == "__main__":
    main()