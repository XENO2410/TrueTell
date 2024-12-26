# src/realtime_processor.py

import asyncio
from typing import List, Dict, Any
from datetime import datetime
from collections import deque
from fact_checker import FactChecker
from source_checker import SourceChecker
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    pipeline,
    BertForSequenceClassification
)
from torch.nn.functional import softmax
import torch
import spacy
import re

class RealTimeProcessor:
    def __init__(self):
        self.buffer = deque(maxlen=1000)
        self.processing_interval = 2
        self.batch_size = 100
        self.is_running = False
        self.callbacks = []
        self.bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Initialize NLP components
        self.classifier = pipeline("zero-shot-classification", 
                                 model="facebook/bart-large-mnli")
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.stop_words = set(stopwords.words('english'))
        self.fact_checker = FactChecker()
        self.source_checker = SourceChecker()

        # Add specialized classification pipelines
        self.stance_detector = pipeline("text-classification", 
                                      model="facebook/bart-large-mnli")
        self.fake_news_detector = pipeline("text-classification", 
                                         model="roberta-base-openai-detector")
        
        # Load SpaCy for better entity recognition
        self.nlp = spacy.load("en_core_web_sm")
               
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

    def _detect_patterns(self, text: str, recent_texts: List[str]) -> Dict:
        """Detect suspicious patterns in content"""
        patterns = {
            'repeated_narratives': self._find_narrative_patterns(text, recent_texts),
            'entity_manipulation': self._detect_entity_manipulation(text),
            'temporal_patterns': self._analyze_temporal_patterns(text),
            'source_patterns': self._analyze_source_patterns(text)
        }
        return patterns

    def _find_narrative_patterns(self, text: str, recent_texts: List[str]) -> Dict:
        """Find repeated narrative patterns"""
        doc = self.nlp(text)
        return {
            'key_phrases': self._extract_key_phrases(doc),
            'narrative_similarity': self._calculate_narrative_similarity(doc, recent_texts),
            'topic_evolution': self._track_topic_evolution(doc)
        }    
        
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
        """Analyze a batch of text items with enhanced ML capabilities"""
        results = []
        recent_texts = [item['text'] for item in list(self.buffer)[-10:]]  # Get recent texts for pattern analysis
        
        for item in batch:
            # Split text into sentences
            sentences = sent_tokenize(item['text'])
            
            for sentence in sentences:
                try:
                    # Enhanced classification analysis
                    classification = self.classifier(
                        sentence,
                        candidate_labels=[
                            "factual statement", 
                            "opinion", 
                            "misleading information",
                            "propaganda",
                            "conspiracy theory"
                        ]
                    )
                    
                    # Enhanced sentiment analysis
                    sentiment = self.sentiment_analyzer(sentence)[0]
                    
                    # Fact checking
                    fact_check_results = self.fact_checker.check_claim(sentence)
                    
                    # Source checking
                    urls = self._extract_urls(sentence)
                    source_checks = [self.source_checker.check_source(url) for url in urls]
                    
                    # Pattern detection
                    patterns = {
                        'narrative_patterns': self._detect_narrative_patterns(sentence, recent_texts),
                        'linguistic_patterns': self._analyze_linguistic_patterns(sentence),
                        'temporal_patterns': self._analyze_temporal_patterns(sentence)
                    }
                    
                    # Calculate enhanced risk score
                    risk_score = self._calculate_risk_score(
                        classification_scores=classification['scores'],
                        fact_check_score=fact_check_results['credibility_score'],
                        sentiment_score=sentiment,
                        text=sentence  # Pass the sentence text for additional analysis
                    )
                    
                    # Create comprehensive analysis
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
                        'patterns_detected': patterns,
                        'confidence_score': self._calculate_confidence(
                            classification['scores'],
                            fact_check_results['credibility_score']
                        ),
                        'processed_at': datetime.now().isoformat()
                    }
                    
                    results.append(analysis)
                    
                except Exception as e:
                    print(f"Error analyzing sentence: {e}")
                    continue
        
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

    def _analyze_linguistic_risk(self, text: str) -> float:
        """Analyze linguistic patterns for risk factors"""
        try:
            # Common misinformation linguistic patterns
            risk_patterns = [
                r"wake up",
                r"they don't want you to know",
                r"secret",
                r"conspiracy",
                r"mainstream media won't tell you",
                r"share before they delete"
            ]
            
            risk_score = 0.0
            for pattern in risk_patterns:
                if re.search(pattern, text.lower()):
                    risk_score += 0.2
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            print(f"Error in linguistic analysis: {e}")
            return 0.0
    
    def _analyze_pattern_risk(self, text: str) -> float:
        """Analyze content patterns for risk factors"""
        try:
            doc = self.nlp(text)
            
            # Risk factors
            risk_score = 0.0
            
            # Check for excessive capitalization
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
            if caps_ratio > 0.3:
                risk_score += 0.2
            
            # Check for excessive punctuation
            punct_ratio = sum(1 for c in text if c in '!?') / len(text) if text else 0
            if punct_ratio > 0.1:
                risk_score += 0.2
            
            # Check for emotional manipulation
            emotional_words = ["shocking", "incredible", "must see", "warning"]
            if any(word in text.lower() for word in emotional_words):
                risk_score += 0.2
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            print(f"Error in pattern analysis: {e}")
            return 0.0
    
    def _calculate_confidence(self, classification_scores: List[float], 
                            fact_check_score: float) -> float:
        """Calculate confidence score for the analysis"""
        try:
            # Average of top classification score and fact check score
            top_classification = max(classification_scores)
            confidence = (top_classification + fact_check_score) / 2
            return confidence
            
        except Exception as e:
            print(f"Error calculating confidence: {e}")
            return 0.5
    
    def _calculate_risk_score(self, classification_scores: List[float], 
                             fact_check_score: float, 
                             sentiment_score: Dict,
                             text: str) -> float:
        """Enhanced risk score calculation with text analysis"""
        try:
            # Get base scores
            misleading_score = classification_scores[2] if len(classification_scores) > 2 else 0
            sentiment_factor = 0.2 if sentiment_score['label'] == 'NEGATIVE' else 0
            
            # Additional risk factors
            linguistic_risk = self._analyze_linguistic_risk(text)
            pattern_risk = self._analyze_pattern_risk(text)
            
            # Weighted combination
            weights = {
                'misleading': 0.4,
                'fact_check': 0.3,
                'sentiment': 0.1,
                'linguistic': 0.1,
                'patterns': 0.1
            }
            
            risk_score = (
                weights['misleading'] * misleading_score +
                weights['fact_check'] * (1 - fact_check_score) +
                weights['sentiment'] * sentiment_factor +
                weights['linguistic'] * linguistic_risk +
                weights['patterns'] * pattern_risk
            )
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            print(f"Error calculating risk score: {e}")
            return 0.5  # Default moderate risk on error

    async def _predict_spread(self, text: str, current_analysis: Dict) -> Dict:
        """Predict potential spread and impact"""
        try:
            features = self._extract_prediction_features(text, current_analysis)
            spread_score = self.spread_predictor.predict(features)
            
            return {
                'spread_probability': float(spread_score),
                'potential_reach': self._estimate_reach(spread_score),
                'virality_factors': self._analyze_virality_factors(text),
                'recommended_actions': self._get_recommended_actions(spread_score)
            }
        except Exception as e:
            print(f"Error in spread prediction: {e}")
            return {}
            
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