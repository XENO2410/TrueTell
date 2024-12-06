# src/app.py
import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import datetime
import numpy as np
import ssl
from fact_checker import FactChecker
from source_checker import SourceChecker
import asyncio
from dotenv import load_dotenv

load_dotenv()

# SSL Certificate fix for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

class MisinformationDetector:
    def __init__(self):
        try:
            # Initialize models
            self.classifier = pipeline("zero-shot-classification", 
                                    model="facebook/bart-large-mnli")
            self.sentiment_analyzer = pipeline("sentiment-analysis")
            self.stop_words = set(stopwords.words('english'))
            self.fact_checker = FactChecker()
            self.source_checker = SourceChecker()
            
        except Exception as e:
            st.error(f"Error initializing models: {e}")
            raise e

    def preprocess_text(self, text):
        try:
            return sent_tokenize(text)
        except Exception as e:
            print(f"Error in preprocess_text: {e}")
            return [text]

    def extract_key_terms(self, text):
        try:
            # Basic tokenization
            tokens = word_tokenize(text)
            
            # Basic filtering without POS tagging
            key_terms = [word for word in tokens 
                        if word.lower() not in self.stop_words
                        and word.isalnum()
                        and len(word) > 2]
            
            return list(set(key_terms))  # Remove duplicates
        except Exception as e:
            print(f"Error in extract_key_terms: {e}")
            return []

    def check_source_credibility(self, text):
        # Simplified credibility scoring
        try:
            # Basic heuristics for credibility
            credibility_score = 0.7  # Default moderate credibility
            
            # Lower credibility for texts with extreme claims
            extreme_phrases = ['never', 'always', 'everyone', 'nobody', 'impossible', 'guaranteed']
            if any(phrase in text.lower() for phrase in extreme_phrases):
                credibility_score *= 0.8
            
            return max(0.1, min(credibility_score, 1.0))  # Keep between 0.1 and 1.0
        except Exception as e:
            print(f"Error in check_source_credibility: {e}")
            return 0.5

    async def analyze_text(self, text):
        if not text:
            return []
            
        try:
            sentences = self.preprocess_text(text)
            results = []
            
            for sentence in sentences:
                try:
                    # Zero-shot classification
                    classification = self.classifier(
                        sentence,
                        candidate_labels=[
                            "factual statement", 
                            "opinion", 
                            "misleading information"
                        ],
                        multi_label=False
                    )
                    
                    # Sentiment analysis
                    sentiment = self.sentiment_analyzer(sentence)
                    
                    # Source credibility
                    source_credibility = self.check_source_credibility(sentence)
                    
                    # Calculate risk score (simplified)
                    risk_score = 1 - classification['scores'][0]
                    
                    # Extract key terms
                    key_terms = self.extract_key_terms(sentence)
                    
                    results.append({
                        'sentence': sentence,
                        'classification': classification['labels'][0],
                        'confidence': classification['scores'][0],
                        'sentiment': sentiment[0],
                        'source_credibility': source_credibility,
                        'risk_score': risk_score,
                        'key_terms': key_terms,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                except Exception as e:
                    print(f"Error processing sentence: {e}")
                    continue
                    
            return results
        except Exception as e:
            print(f"Error in analyze_text: {e}")
            return []

def create_analysis_charts(results):
    try:
        # Create timeline DataFrame
        df_timeline = pd.DataFrame([
            {
                'timestamp': r['timestamp'],
                'risk_score': r['risk_score']
            } for r in results
        ])
        
        # Risk timeline
        fig_timeline = px.line(
            df_timeline,
            x='timestamp',
            y='risk_score',
            title='Risk Score Timeline'
        )
        
        # Classification distribution
        classifications = [r['classification'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        fig_dist = px.bar(
            x=classifications,
            y=confidences,
            title='Classification Distribution'
        )
        
        return fig_timeline, fig_dist
    except Exception as e:
        print(f"Error creating charts: {e}")
        return None, None

def display_live_results(results):
    try:
        if not results:
            st.warning("No results to display")
            return
            
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        avg_risk = np.mean([r.get('risk_score', 0) for r in results])
        avg_credibility = np.mean([r.get('source_credibility', 0) for r in results])
        
        with col1:
            st.metric("Average Risk Score", f"{avg_risk:.2%}")
        
        with col2:
            st.metric("Source Credibility", f"{avg_credibility:.2%}")
        
        with col3:
            high_risk_count = sum(1 for r in results if r.get('risk_score', 0) > 0.7)
            st.metric("High Risk Statements", high_risk_count)
        
        # Display charts
        charts = create_analysis_charts(results)
        if charts and charts[0] and charts[1]:
            st.plotly_chart(charts[0])
            st.plotly_chart(charts[1])
            
    except Exception as e:
        st.error(f"Error displaying results: {e}")

def text_analysis_tab():
    st.title("🔍 Real-time Misinformation Detector")
    
    # Input section
    st.header("Input Text")
    text_input = st.text_area(
        "Enter text to analyze:",
        height=150
    )
    
    if st.button("Analyze Text"):
        if text_input:
            with st.spinner("Analyzing text..."):
                try:
                    # Process the text
                    results = st.session_state.detector.analyze_text(text_input)
                    
                    if results:
                        # Display results
                        st.header("Analysis Results")
                        
                        for result in results:
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.markdown(f"**Sentence:** {result['sentence']}")
                                    st.markdown(f"**Classification:** {result['classification']}")
                                    st.markdown(f"**Key Terms:** {', '.join(result['key_terms'])}")
                                    st.markdown(f"**Sentiment:** {result['sentiment']['label']} "
                                              f"({result['sentiment']['score']:.2%})")
                                
                                with col2:
                                    st.metric("Risk Score", f"{result['risk_score']:.2%}")
                                    st.metric("Source Credibility", 
                                            f"{result['source_credibility']:.2%}")
                                
                                if result['risk_score'] > 0.7:
                                    st.warning("⚠️ High risk of misinformation!")
                                
                                st.divider()
                        
                        # Display overall analysis
                        display_live_results(results)
                    else:
                        st.warning("No results were generated. Please try again.")
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
        else:
            st.warning("Please enter some text to analyze.")

def live_monitor_tab():
    st.title("📡 Live Broadcast Monitor")
    
    source_type = st.selectbox(
        "Select Source",
        ["Live News Feed", "Social Media Stream", "Custom Input"]
    )
    
    if st.button("Start Monitoring"):
        placeholder = st.empty()
        
        try:
            # Simulated live feed
            with st.spinner("Monitoring live feed..."):
                results = []
                for i in range(5):  # Simulate 5 updates
                    # Simulate incoming text
                    live_text = f"This is a simulated live feed message {i+1}. " \
                              f"{'Some misleading content.' if i % 2 else 'Factual content.'}"
                    
                    # Analyze text
                    new_results = st.session_state.detector.analyze_text(live_text)
                    if new_results:
                        results.extend(new_results)
                        
                        # Update display
                        with placeholder.container():
                            display_live_results(results)
                    
                    time.sleep(2)  # Simulate delay between updates
        except Exception as e:
            st.error(f"Error in live monitoring: {e}")

def main():
    st.set_page_config(
        page_title="Real-time Misinformation Detector",
        layout="wide"
    )
    
    # Initialize detector
    if 'detector' not in st.session_state:
        try:
            st.session_state.detector = MisinformationDetector()
        except Exception as e:
            st.error(f"Failed to initialize the detector: {e}")
            return
    
    # Create tabs
    tab1, tab2 = st.tabs(["Text Analysis", "Live Monitor"])
    
    with tab1:
        text_analysis_tab()
    
    with tab2:
        live_monitor_tab()

if __name__ == "__main__":
    main()