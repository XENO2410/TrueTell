# src/fact_checker.py
from typing import Dict, Optional, List
import requests
from datetime import datetime
import os
from credibility_scorer import CredibilityScorer

class FactChecker:
    def __init__(self):
        self.google_api_key = os.getenv('GOOGLE_FACT_CHECK_API_KEY')
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.credible_domains = self._load_credible_domains()
        self.scorer = CredibilityScorer()

    def _load_credible_domains(self) -> Dict[str, float]:
        return {
            'reuters.com': 0.95,
            'apnews.com': 0.95,
            'bbc.com': 0.90,
            'nytimes.com': 0.85,
            'theguardian.com': 0.85,
        }

    def check_claim(self, claim: str, classification_scores: List[float] = None, 
                   sentiment_score: Dict = None) -> Dict:
        # Default scores if not provided
        if classification_scores is None:
            classification_scores = [0.5, 0.3, 0.2]
        if sentiment_score is None:
            sentiment_score = {'label': 'NEUTRAL', 'score': 0.5}

        # Get credibility analysis
        credibility_analysis = self.scorer.calculate_credibility_score(
            claim, classification_scores, sentiment_score
        )

        results = {
            'claim': claim,
            'matches': self._check_against_database(claim),
            'credibility_score': credibility_analysis['final_score'],
            'credibility_analysis': credibility_analysis,
            'timestamp': datetime.now().isoformat()
        }
        return results

    def _check_against_database(self, claim: str) -> list:
        return [{
            'text': claim,
            'rating': 'Unverified',
            'source': 'Internal Database'
        }]