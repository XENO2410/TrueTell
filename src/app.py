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
from utils import download_nltk_data
from fact_checker import FactChecker
from source_checker import SourceChecker
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
            
            # Basic filtering without POS tagging
            basic_terms = [word for word in tokens 
                          if word.isalnum() 
                          and word.lower() not in self.stop_words
                          and len(word) > 2]
            
            try:
                # Try POS tagging
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
            # Classification analysis
            classification = self.classifier(
                sentence,
                candidate_labels=[
                    "factual statement", 
                    "opinion", 
                    "misleading information"
                ]
            )
            
            # Sentiment analysis
            sentiment = self.sentiment_analyzer(sentence)[0]
            
            # Fact checking
            fact_check_results = self.fact_checker.check_claim(sentence)
            
            # Source checking for URLs
            urls = self._extract_urls(sentence)
            source_checks = [self.source_checker.check_source(url) for url in urls]
            
            # Calculate risk score
            risk_score = self.calculate_risk_score(
                classification['scores'],
                fact_check_results['credibility_score'],
                sentiment
            )
            
            # Extract key terms
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
    # Create risk timeline
    df = pd.DataFrame(results)
    fig_timeline = px.line(
        df,
        x='timestamp',
        y='risk_score',
        title='Risk Score Timeline'
    )
    
    # Create classification distribution
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
    
    fig_timeline, fig_dist = create_analysis_charts(results)
    st.plotly_chart(fig_timeline)
    st.plotly_chart(fig_dist)

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
    
    if source_type == "Live News Feed":
        placeholder = st.empty()
        
        if st.button("Start Monitoring"):
            with st.spinner("Monitoring live feed..."):
                results = []
                for i in range(5):
                    live_text = f"This is a simulated live feed message {i+1}. " \
                              f"{'Some misleading content.' if i % 2 else 'Factual content.'}"
                    
                    new_results = st.session_state.detector.analyze_text(live_text)
                    results.extend(new_results)
                    
                    with placeholder.container():
                        display_live_results(results)
                    
                    time.sleep(2)

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