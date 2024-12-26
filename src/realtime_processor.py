# src/realtime_processor.py

import asyncio
from typing import List, Dict, Any, Optional
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
from functools import lru_cache
import torch.cuda
import time
from dataclasses import dataclass
import logging
import traceback
import psutil
import os

@dataclass
class PerformanceMetrics:
    processing_time: float
    memory_usage: float
    model_latency: Dict[str, float]
    batch_size: int
    error_count: int
    
class RealTimeProcessor:
    MODEL_VERSIONS = {
        'bert': 'bert-base-uncased-v1',
        'roberta': 'roberta-base-v1',
        'sentiment': 'nlptown/bert-base-multilingual-uncased-sentiment-v1'        
    }
    def __init__(self):
        self.buffer = deque(maxlen=1000)
        self.processing_interval = 2
        self.batch_size = 100
        self.is_running = False
        self.callbacks = []
        self.bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models = {}
        self.model_cache = {}
        self.initialize_models()
        self.metrics = []  
        self._setup_logging()
                      
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

        self.emotional_lexicon = set([
            "angry", "sad", "happy", "excited", "shocking",
            "incredible", "amazing", "terrible", "wonderful",
            "horrible", "outrageous", "unbelievable"
        ])
        
    def _setup_logging(self):
        """Setup detailed logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('truthtell.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('TruthTell')
    
    def _log_error(self, message: str, error: Exception, level: str = 'error'):
        """Enhanced error logging"""
        error_details = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'error_type': type(error).__name__,
            'error_details': str(error),
            'stack_trace': traceback.format_exc()
        }
        
        if level == 'error':
            self.logger.error(error_details)
        elif level == 'warning':
            self.logger.warning(error_details)
            
    def initialize_models(self):
        """Initialize models with version control and caching"""
        try:
            # Cache models in memory
            self.models['bert'] = self._load_model_cached('bert')
            self.models['roberta'] = self._load_model_cached('roberta')
            print(f"Models loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    @lru_cache(maxsize=3)  # Cache up to 3 models
    def _load_model_cached(self, model_name: str):
        """Load and cache models with version control"""
        version = self.MODEL_VERSIONS.get(model_name)
        if not version:
            raise ValueError(f"Unknown model: {model_name}")
        return AutoModelForSequenceClassification.from_pretrained(version).to(self.device)
                   
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

    def _get_memory_usage(self) -> float:
        """Get current memory usage of the process"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    def _check_performance_threshold(self):
        """Check if performance metrics exceed thresholds"""
        if len(self.metrics) < 2:
            return
            
        latest = self.metrics[-1]
        if latest.processing_time > 5.0:  # More than 5 seconds
            self._log_error(
                "Performance degradation detected", 
                Exception(f"Processing time: {latest.processing_time}s"), 
                level='warning'
            )
    
    def _detect_entity_manipulation(self, text: str) -> Dict:
        """Detect potential manipulation of entity information"""
        doc = self.nlp(text)
        entities = {ent.text: ent.label_ for ent in doc.ents}
        return {
            'entities': entities,
            'suspicious_patterns': self._check_entity_patterns(entities)
        }
    
    def _analyze_temporal_patterns(self, text: str) -> Dict:
        """Analyze temporal patterns in content"""
        doc = self.nlp(text)
        return {
            'temporal_references': self._extract_temporal_references(doc),
            'sequence_patterns': self._analyze_sequence_patterns(doc)
        }
    
    def _analyze_source_patterns(self, text: str) -> Dict:
        """Analyze patterns related to content sources"""
        urls = self._extract_urls(text)
        return {
            'source_count': len(urls),
            'source_types': self._categorize_sources(urls),
            'source_reliability': self._check_source_reliability(urls)
        }
    
    def _extract_temporal_references(self, doc) -> List[str]:
        """Extract temporal references from text"""
        temporal_refs = []
        for ent in doc.ents:
            if ent.label_ in ['DATE', 'TIME']:
                temporal_refs.append(ent.text)
        return temporal_refs
    
    def _analyze_sequence_patterns(self, doc) -> Dict:
        """Analyze sequence patterns in content"""
        return {
            'narrative_flow': self._analyze_narrative_flow(doc),
            'temporal_sequence': self._check_temporal_sequence(doc)
        }
    
    def _analyze_narrative_flow(self, doc) -> str:
        """Analyze the narrative flow of content"""
        # Basic implementation
        return "sequential" if len(list(doc.sents)) > 1 else "single"
    
    def _check_temporal_sequence(self, doc) -> bool:
        """Check if temporal sequence makes logical sense"""
        # Basic implementation
        return True
    
    def _check_entity_patterns(self, entities: Dict) -> List[str]:
        """Check for suspicious patterns in entity usage"""
        suspicious_patterns = []
        # Add pattern detection logic here
        return suspicious_patterns
    
    def _categorize_sources(self, urls: List[str]) -> Dict[str, int]:
        """Categorize sources by type"""
        categories = {'news': 0, 'social': 0, 'blog': 0, 'other': 0}
        for url in urls:
            # Add categorization logic here
            categories['other'] += 1
        return categories
    
    def _check_source_reliability(self, urls: List[str]) -> Dict[str, float]:
        """Check reliability scores for sources"""
        return {url: self.source_checker.check_source(url).get('reliability', 0.0) 
                for url in urls}
    
    def _analyze_linguistic_patterns(self, text: str) -> Dict:
        """Analyze linguistic patterns in text"""
        return {
            'sentiment_patterns': self._analyze_sentiment_patterns(text),
            'rhetorical_devices': self._detect_rhetorical_devices(text),
            'language_complexity': self._analyze_language_complexity(text)
        }
    
    def _analyze_sentiment_patterns(self, text: str) -> Dict:
        """Analyze sentiment patterns"""
        return {
            'overall_sentiment': self.sentiment_analyzer(text)[0],
            'emotional_intensity': self._calculate_emotional_intensity(text)
        }
    
    def _calculate_emotional_intensity(self, text: str) -> float:
        """Calculate emotional intensity of text"""
        # Basic implementation
        emotional_words = len([word for word in text.lower().split() 
                              if word in self.emotional_lexicon])
        return min(emotional_words / len(text.split()), 1.0)
       
    async def process_batch(self):
        """Optimized batch processing"""
        if len(self.buffer) < self.batch_size:
            return
            
        batch = []
        batch_size = 0
        max_batch_memory = 1024 * 1024 * 512  # 512MB limit
        
        while self.buffer and batch_size < max_batch_memory:
            item = self.buffer.popleft()
            estimated_size = len(str(item)) * 2  # Rough size estimation
            if batch_size + estimated_size > max_batch_memory:
                break
                
            batch.append(item)
            batch_size += estimated_size
        
        if batch:
            try:
                results = await self._analyze_batch(batch)
                await self._notify_callbacks(results)
            except Exception as e:
                self._log_error("Batch processing failed", e)
                # Return items to buffer
                self.buffer.extendleft(reversed(batch))
    
    async def _analyze_batch(self, batch: List[Dict]) -> List[Dict]:
        start_time = time.time()
        error_count = 0
        model_times = {}
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

                    self.metrics.append(PerformanceMetrics(
                        processing_time=time.time() - start_time,
                        memory_usage=self._get_memory_usage(),
                        model_latency=model_times,
                        batch_size=len(batch),
                        error_count=error_count
                    ))
                    
                    # Log if performance degrades
                    self._check_performance_threshold()
                    
                except Exception as e:
                    error_count += 1
                    self._log_error("Batch analysis error", e)                    
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