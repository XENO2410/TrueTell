# src/processor.py
from transformers import pipeline
import re

class TruthTellProcessor:
    def __init__(self, model_name, confidence_threshold):
        self.fact_checker = pipeline("text-classification", 
                                   model=model_name)
        self.confidence_threshold = confidence_threshold
        
    def process_text(self, text):
        # Simple sentence splitting using regular expressions
        sentences = re.split('[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        results = []
        
        for sentence in sentences:
            fact_check_result = self.fact_check(sentence)
            results.append({
                'text': sentence,
                'confidence': fact_check_result['score'],
                'classification': fact_check_result['label'],
                'sources': self.get_sources(sentence)
            })
            
        return results
    
    def fact_check(self, text):
        result = self.fact_checker(text, 
                                 candidate_labels=['true', 
                                                 'false', 
                                                 'unverified'])
        return result
    
    def get_sources(self, text):
        # Implement actual API calls here
        return ["Reuters Fact Check", "Official Government Data"]