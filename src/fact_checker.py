# src/fact_checker.py
from typing import Dict, Optional, List
import requests
from datetime import datetime
import os
from credibility_scorer import CredibilityScorer
import pandas as pd
from difflib import get_close_matches

class FactChecker:
    def __init__(self):
        self.google_api_key = os.getenv('GOOGLE_FACT_CHECK_API_KEY')
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.credible_domains = self._load_credible_domains()
        self.scorer = CredibilityScorer()
        self.fact_data = self._load_fact_dataset()
        self.fact_cache = {}

    def _load_credible_domains(self) -> Dict[str, float]:
        return {
            'reuters.com': 0.95,
            'apnews.com': 0.95,
            'bbc.com': 0.90,
            'nytimes.com': 0.85,
            'theguardian.com': 0.85,
        }

    def _load_fact_dataset(self) -> Dict:
        """Load and structure the fact-checking dataset"""
        try:
            df = pd.read_csv('datasets/factdata.csv')
            
            # Create structured data for quick lookup
            fact_data = {
                'verified_true': df[df['Fact Check'] == 1]['Claims'].tolist(),
                'verified_false': df[df['Fact Check'] == 0]['Claims'].tolist(),
                'unverified': df[df['Fact Check'] == 2]['Claims'].tolist(),
                'lookup_table': dict(zip(df['Claims'], df['Fact Check'])),
                'total_claims': len(df)
            }
            
            print(f"Loaded {len(df)} fact-check entries")
            return fact_data
            
        except Exception as e:
            print(f"Error loading fact dataset: {e}")
            return {
                'verified_true': [],
                'verified_false': [],
                'unverified': [],
                'lookup_table': {},
                'total_claims': 0
            }

    def check_claim(self, claim: str, classification_scores: List[float] = None, 
                   sentiment_score: Dict = None) -> Dict:
        """Enhanced claim checking with dataset integration"""
        # Check cache first
        if claim in self.fact_cache:
            return self.fact_cache[claim]

        # Default scores if not provided
        if classification_scores is None:
            classification_scores = [0.5, 0.3, 0.2]
        if sentiment_score is None:
            sentiment_score = {'label': 'NEUTRAL', 'score': 0.5}

        # Get credibility analysis
        credibility_analysis = self.scorer.calculate_credibility_score(
            claim, classification_scores, sentiment_score
        )
        
        # Check against dataset
        dataset_check = self._check_against_database(claim)
        
        # Combine results
        results = {
            'claim': claim,
            'matches': dataset_check['matches'],
            'credibility_score': self._combine_scores(
                credibility_analysis['final_score'],
                dataset_check['confidence']
            ),
            'credibility_analysis': {
                **credibility_analysis,
                'dataset_match': dataset_check['match_type'],
                'dataset_confidence': dataset_check['confidence']
            },
            'similar_claims': self._find_similar_claims(claim),
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache the result
        self.fact_cache[claim] = results
        return results

    def _check_against_database(self, claim: str) -> Dict:
        """Check claim against the fact-checking dataset"""
        # Look for exact matches
        if claim in self.fact_data['lookup_table']:
            status_map = {0: 'false', 1: 'true', 2: 'unverified'}
            return {
                'matches': [{
                    'text': claim,
                    'rating': status_map[self.fact_data['lookup_table'][claim]],
                    'source': 'Fact Database',
                    'confidence': 1.0
                }],
                'match_type': 'exact',
                'confidence': 1.0
            }
        
        # Look for similar claims
        similar_claims = get_close_matches(
            claim, 
            list(self.fact_data['lookup_table'].keys()), 
            n=3, 
            cutoff=0.6
        )
        
        if similar_claims:
            matches = []
            for similar_claim in similar_claims:
                status_map = {0: 'false', 1: 'true', 2: 'unverified'}
                similarity_score = self._calculate_similarity(claim, similar_claim)
                matches.append({
                    'text': similar_claim,
                    'rating': status_map[self.fact_data['lookup_table'][similar_claim]],
                    'source': 'Fact Database',
                    'confidence': similarity_score
                })
            
            return {
                'matches': matches,
                'match_type': 'similar',
                'confidence': matches[0]['confidence'] if matches else 0.0
            }
        
        return {
            'matches': [{
                'text': claim,
                'rating': 'Unverified',
                'source': 'Internal Database',
                'confidence': 0.0
            }],
            'match_type': 'none',
            'confidence': 0.0
        }

    def _find_similar_claims(self, claim: str) -> List[Dict]:
        """Find similar claims from the dataset"""
        similar_claims = get_close_matches(
            claim, 
            list(self.fact_data['lookup_table'].keys()), 
            n=5, 
            cutoff=0.6
        )
        
        return [{
            'claim': similar_claim,
            'status': self.fact_data['lookup_table'][similar_claim],
            'similarity': self._calculate_similarity(claim, similar_claim)
        } for similar_claim in similar_claims]

    def _calculate_similarity(self, claim1: str, claim2: str) -> float:
        """Calculate similarity between two claims"""
        # Simple word overlap similarity
        words1 = set(claim1.lower().split())
        words2 = set(claim2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

    def _combine_scores(self, credibility_score: float, dataset_confidence: float) -> float:
        """Combine credibility score with dataset confidence"""
        # Weight the scores (adjust weights as needed)
        credibility_weight = 0.7
        dataset_weight = 0.3
        
        combined_score = (
            credibility_score * credibility_weight +
            dataset_confidence * dataset_weight
        )
        
        return min(1.0, max(0.0, combined_score))

    def get_statistics(self) -> Dict:
        """Get fact-checking statistics"""
        return {
            'total_claims': self.fact_data['total_claims'],
            'verified_true': len(self.fact_data['verified_true']),
            'verified_false': len(self.fact_data['verified_false']),
            'unverified': len(self.fact_data['unverified']),
            'cache_size': len(self.fact_cache)
        }