# src/fact_checker.py
from typing import Dict, Optional
import requests
from datetime import datetime
import os

class FactChecker:
    def __init__(self):
        self.google_api_key = os.getenv('GOOGLE_FACT_CHECK_API_KEY')
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.credible_domains = self._load_credible_domains()

    def _load_credible_domains(self) -> Dict[str, float]:
        return {
            'reuters.com': 0.95,
            'apnews.com': 0.95,
            'bbc.com': 0.90,
            'nytimes.com': 0.85,
            'theguardian.com': 0.85,
        }

    def check_claim(self, claim: str) -> Dict:
        results = {
            'claim': claim,
            'matches': self._check_against_database(claim),
            'credibility_score': self._calculate_credibility_score(claim),
            'timestamp': datetime.now().isoformat()
        }
        return results

    def _check_against_database(self, claim: str) -> list:
        return [{
            'text': claim,
            'rating': 'Unverified',
            'source': 'Internal Database'
        }]

    def _calculate_credibility_score(self, text: str) -> float:
        score = 0.5

        for domain, trust_score in self.credible_domains.items():
            if domain in text.lower():
                score += trust_score * 0.3

        if '"' in text or '"' in text:
            score += 0.1
        if any(word in text.lower() for word in ['according to', 'study shows', 'research']):
            score += 0.1

        return min(score, 1.0)