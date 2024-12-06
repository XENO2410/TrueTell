# src/realtime_processor.py

import asyncio
from typing import List, Dict, Any
from datetime import datetime
from collections import deque
from fact_checker import FactChecker
from source_checker import SourceChecker
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

class RealTimeProcessor:
    def __init__(self):
        self.buffer = deque(maxlen=1000)
        self.processing_interval = 2
        self.batch_size = 100
        self.is_running = False
        self.callbacks = []
        
        # Initialize NLP components
        self.classifier = pipeline("zero-shot-classification", 
                                 model="facebook/bart-large-mnli")
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.stop_words = set(stopwords.words('english'))
        self.fact_checker = FactChecker()
        self.source_checker = SourceChecker()
        
    async def start(self):
        """Start the real-time processing loop"""
        self.is_running = True
        while self.is_running:
            if len(self.buffer) >= self.batch_size:
                await self.process_batch()
            await asyncio.sleep(self.processing_interval)
    
    def add_text(self, text: str, source: str = None):
        """Add new text to the processing buffer"""
        timestamp = datetime.now().isoformat()
        self.buffer.append({
            'text': text,
            'source': source,
            'timestamp': timestamp,
            'processed': False
        })
    
    async def process_batch(self):
        """Process a batch of text from the buffer"""
        batch = []
        for _ in range(self.batch_size):
            if not self.buffer:
                break
            item = self.buffer.popleft()
            if not item['processed']:
                batch.append(item)
        
        if batch:
            results = await self._analyze_batch(batch)
            await self._notify_callbacks(results)
    
    async def _analyze_batch(self, batch: List[Dict]) -> List[Dict]:
        """Analyze a batch of text items"""
        results = []
        for item in batch:
            # Split text into sentences
            sentences = sent_tokenize(item['text'])
            
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
                
                # Source checking
                urls = self._extract_urls(sentence)
                source_checks = [self.source_checker.check_source(url) for url in urls]
                
                # Calculate risk score
                risk_score = self._calculate_risk_score(
                    classification['scores'],
                    fact_check_results['credibility_score'],
                    sentiment
                )
                
                analysis = {
                    'timestamp': item['timestamp'],
                    'source': item['source'],
                    'sentence': sentence,
                    'classifications': classification['labels'],
                    'classification_scores': classification['scores'],
                    'sentiment': sentiment,
                    'fact_check_results': fact_check_results,
                    'source_checks': source_checks,
                    'urls': urls,
                    'risk_score': risk_score,
                    'key_terms': self._extract_key_terms(sentence),
                    'processed_at': datetime.now().isoformat()
                }
                results.append(analysis)
        
        return results
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        words = text.split()
        return [word for word in words if word.startswith(('http://', 'https://'))]
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
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
    
    def _calculate_risk_score(self, classification_scores: List[float], 
                            fact_check_score: float, 
                            sentiment_score: Dict) -> float:
        """Calculate risk score based on various factors"""
        misleading_score = classification_scores[2] if len(classification_scores) > 2 else 0
        sentiment_factor = 0.2 if sentiment_score['label'] == 'NEGATIVE' else 0
        
        risk_score = (misleading_score * 0.5 + 
                     (1 - fact_check_score) * 0.3 + 
                     sentiment_factor)
        
        return min(risk_score, 1.0)
    
    def add_callback(self, callback):
        """Add a callback function to be notified of results"""
        self.callbacks.append(callback)
    
    async def _notify_callbacks(self, results: List[Dict]):
        """Notify all registered callbacks with results"""
        for callback in self.callbacks:
            try:
                await callback(results)
            except Exception as e:
                print(f"Error in callback: {e}")
    
    def stop(self):
        """Stop the processing loop"""
        self.is_running = False