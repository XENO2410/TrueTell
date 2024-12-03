# src/app.py
import streamlit as st
from processor import TruthTellProcessor
from knowledge_graph import KnowledgeGraph
from config import Config

def initialize_session_state():
    if 'processor' not in st.session_state:
        st.session_state.processor = TruthTellProcessor(
            Config.MODEL_NAME,
            Config.CONFIDENCE_THRESHOLD
        )
    if 'knowledge_graph' not in st.session_state:
        st.session_state.knowledge_graph = KnowledgeGraph()

def create_dashboard():
    st.title("TruthTell Real-time Fact Checker")
    
    # Sidebar
    st.sidebar.title("Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        0.0, 1.0, 0.75
    )
    
    # Main content
    st.header("Live Broadcast Monitor")
    input_text = st.text_area(
        "Enter broadcast text or connect to live feed:"
    )
    
    if st.button("Analyze"):
        if input_text:
            results = st.session_state.processor.process_text(input_text)
            
            st.header("Analysis Results")
            
            for result in results:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(result['text'])
                        st.write(f"Classification: {result['classification']}")
                    
                    with col2:
                        confidence = result['confidence']
                        st.progress(confidence)
                        st.write(f"Confidence: {confidence:.2%}")
                    
                    st.write("Sources:", ", ".join(result['sources']))
                    st.divider()
                    
                    # Store in knowledge graph
                    st.session_state.knowledge_graph.add_fact(
                        result['text'],
                        result['classification'],
                        result['sources']
                    )

def main():
    initialize_session_state()
    create_dashboard()

if __name__ == "__main__":
    main()