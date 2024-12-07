# src/credibility_scorer.py
from typing import Dict, List
from datetime import datetime
import re
from textblob import TextBlob
import numpy as np

class CredibilityScorer:
    def __init__(self):
        self.weights = {
            'content_quality': 0.3,
            'source_credibility': 0.25,
            'claim_characteristics': 0.25,
            'sentiment_impact': 0.2
        }
        
        self.credible_phrases = [
            'according to', 'research shows', 'study finds',
            'evidence suggests', 'data indicates', 'experts say'
        ]
        
        self.suspicious_phrases = [
            'they don\'t want you to know', 'secret cure',
            'miracle solution', 'shocking truth', '100% guaranteed',
            'they lied to you'
        ]
        
        self.claim_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict:
        return {
            'urls': re.compile(
                r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|'
                r'(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            ),
            'numbers': re.compile(r'\b\d+(?:\.\d+)?%?\b'),
            'dates': re.compile(
                r'\b\d{4}\b|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|'
                r'Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|'
                r'Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b'
            ),
            'emotional': re.compile(
                r'\b(?:shocking|incredible|amazing|outrageous|must[- ]see)\b',
                re.IGNORECASE
            ),
            'certainty': re.compile(
                r'\b(?:absolutely|definitely|always|never|everyone|nobody)\b',
                re.IGNORECASE
            )
        }

    def calculate_credibility_score(self, text: str, 
                                  classification_scores: List[float],
                                  sentiment_score: Dict) -> Dict:
        """Calculate comprehensive credibility score"""
        # Calculate component scores
        content_score = self._analyze_content_quality(text)
        source_score = self._analyze_source_credibility(text)
        claim_score = self._analyze_claim_characteristics(
            text, classification_scores
        )
        sentiment_score = self._analyze_sentiment_impact(sentiment_score)
        
        # Calculate weighted final score
        final_score = (
            content_score * self.weights['content_quality'] +
            source_score * self.weights['source_credibility'] +
            claim_score * self.weights['claim_characteristics'] +
            sentiment_score * self.weights['sentiment_impact']
        )
        
        return {
            'final_score': final_score,
            'components': {
                'content_quality': content_score,
                'source_credibility': source_score,
                'claim_characteristics': claim_score,
                'sentiment_impact': sentiment_score
            },
            'analysis': self._generate_analysis(text, final_score),
            'flags': self._generate_flags(text),
            'timestamp': datetime.now().isoformat()
        }

    def _analyze_content_quality(self, text: str) -> float:
        """Analyze the quality of content"""
        blob = TextBlob(text)
        
        # Check for specific indicators
        has_numbers = bool(self.claim_patterns['numbers'].search(text))
        has_dates = bool(self.claim_patterns['dates'].search(text))
        has_urls = bool(self.claim_patterns['urls'].search(text))
        
        # Calculate base scores
        specificity_score = sum([has_numbers, has_dates, has_urls]) / 3
        language_score = min(1.0, len(blob.words) / 100)
        
        # Check for credible phrases
        credible_phrase_score = sum(
            1 for phrase in self.credible_phrases if phrase in text.lower()
        ) / len(self.credible_phrases)
        
        return np.mean([specificity_score, language_score, credible_phrase_score])

    def _analyze_source_credibility(self, text: str) -> float:
        """Analyze source credibility"""
        # Extract URLs
        urls = self.claim_patterns['urls'].findall(text)
        if not urls:
            return 0.5  # Neutral score for no sources
        
        # Basic source analysis (can be expanded)
        score = 0.5
        for url in urls:
            if any(domain in url.lower() for domain in [
                'gov', 'edu', 'org', 'reuters.com', 'ap.org', 
                'bbc.com', 'nature.com', 'science.org'
            ]):
                score += 0.1
            if 'wikipedia.org' in url.lower():
                score += 0.05
                
        return min(1.0, score)

    def _analyze_claim_characteristics(self, text: str, 
                                     classification_scores: List[float]) -> float:
        """Analyze characteristics of the claim"""
        # Check for suspicious patterns
        emotional_language = bool(self.claim_patterns['emotional'].search(text))
        absolute_claims = bool(self.claim_patterns['certainty'].search(text))
        
        suspicious_phrase_score = sum(
            1 for phrase in self.suspicious_phrases if phrase in text.lower()
        ) / len(self.suspicious_phrases)
        
        # Use classification scores
        factual_score = classification_scores[0]  # Assuming first score is factual
        
        pattern_score = 1.0 - (
            (emotional_language + absolute_claims + suspicious_phrase_score) / 3
        )
        
        return np.mean([pattern_score, factual_score])

    def _analyze_sentiment_impact(self, sentiment_score: Dict) -> float:
        """Analyze impact of sentiment on credibility"""
        # Convert negative sentiment to lower credibility
        if sentiment_score['label'] == 'NEGATIVE':
            return 0.3 + (1 - sentiment_score['score']) * 0.4
        return 0.7 + sentiment_score['score'] * 0.3

    def _generate_analysis(self, text: str, final_score: float) -> Dict:
        """Generate detailed analysis"""
        severity = 'Low' if final_score >= 0.7 else \
                  'Medium' if final_score >= 0.4 else 'High'
                  
        return {
            'risk_level': severity,
            'summary': self._generate_summary(final_score),
            'recommendations': self._generate_recommendations(text, final_score)
        }

    def _generate_summary(self, score: float) -> str:
        """Generate summary based on score"""
        if score >= 0.8:
            return "Highly credible content with strong supporting evidence"
        elif score >= 0.6:
            return "Moderately credible content with some supporting evidence"
        elif score >= 0.4:
            return "Content requires additional verification"
        elif score >= 0.2:
            return "Content shows several signs of potential misinformation"
        else:
            return "Content has multiple red flags for misinformation"

    def _generate_recommendations(self, text: str, score: float) -> List[str]:
        """Generate recommendations based on analysis"""
        recs = []
        
        if not self.claim_patterns['urls'].search(text):
            recs.append("Include sources to support claims")
        
        if self.claim_patterns['emotional'].search(text):
            recs.append("Reduce emotional language for more objective content")
            
        if self.claim_patterns['certainty'].search(text):
            recs.append("Avoid absolute claims without strong evidence")
            
        if score < 0.4:
            recs.append("Seek additional verification from reliable sources")
            
        return recs

    def _generate_flags(self, text: str) -> List[str]:
        """Generate warning flags for suspicious content"""
        flags = []
        
        if self.claim_patterns['emotional'].search(text):
            flags.append("Emotional manipulation detected")
            
        if self.claim_patterns['certainty'].search(text):
            flags.append("Absolute claims without sufficient evidence")
            
        if any(phrase in text.lower() for phrase in self.suspicious_phrases):
            flags.append("Suspicious narrative patterns detected")
            
        return flags