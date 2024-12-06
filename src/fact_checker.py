# fact_checker.py
import requests
from typing import Dict, List, Optional
import os
from datetime import datetime

class FactChecker:
    def __init__(self):
        # Initialize with multiple fact-checking APIs
        self.google_api_key = os.getenv('GOOGLE_FACT_CHECK_API_KEY')
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.credible_domains = self._load_credible_domains()

    def _load_credible_domains(self) -> Dict[str, float]:
        # This could be expanded to load from a database
        return {
            'reuters.com': 0.95,
            'apnews.com': 0.95,
            'bbc.com': 0.90,
            'nytimes.com': 0.85,
            'theguardian.com': 0.85,
            # Add more trusted sources
        }

    async def check_claim(self, claim: str) -> Dict:
        """Check a claim against multiple fact-checking sources"""
        results = {
            'google_fact_check': await self._check_google_fact_api(claim),
            'news_api': await self._check_news_api(claim),
            'credibility_score': self._calculate_credibility_score(claim),
            'timestamp': datetime.now().isoformat()
        }
        return results

    async def _check_google_fact_api(self, claim: str) -> Dict:
        """Check claim using Google's Fact Check API"""
        try:
            url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
            params = {
                'key': self.google_api_key,
                'query': claim
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            return {'error': 'API request failed'}
        except Exception as e:
            return {'error': str(e)}

    async def _check_news_api(self, claim: str) -> Dict:
        """Cross-reference claim with recent news articles"""
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'apiKey': self.news_api_key,
                'q': claim,
                'sortBy': 'relevancy',
                'language': 'en'
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            return {'error': 'API request failed'}
        except Exception as e:
            return {'error': str(e)}

    def _calculate_credibility_score(self, text: str) -> float:
        """Calculate credibility score based on various factors"""
        # Implement sophisticated credibility scoring
        score = 0.5  # Base score
        
        # Check for credible sources mentioned
        for domain, trust_score in self.credible_domains.items():
            if domain in text.lower():
                score += trust_score * 0.3

        # Check for citation patterns
        if '"' in text or '"' in text:
            score += 0.1
        if any(word in text.lower() for word in ['according to', 'study shows', 'research']):
            score += 0.1

        return min(score, 1.0)