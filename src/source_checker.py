# src/source_checker.py
from typing import Dict, Optional, Set
from urllib.parse import urlparse
from datetime import datetime
import requests
from bs4 import BeautifulSoup

class SourceChecker:
    def __init__(self):
        self.cache: Dict[str, Dict] = {}
        self.blacklist: Set[str] = self._load_blacklist()
        self.whitelist: Set[str] = self._load_whitelist()

    def _load_blacklist(self) -> Set[str]:
        return {
            'fake-news.com',
            'conspiracy-daily.com',
        }

    def _load_whitelist(self) -> Set[str]:
        return {
            'reuters.com',
            'apnews.com',
            'bbc.com',
        }

    def check_source(self, url: str) -> Dict:
        if url in self.cache:
            return self.cache[url]

        domain = urlparse(url).netloc
        
        results = {
            'domain': domain,
            'blacklisted': domain in self.blacklist,
            'whitelisted': domain in self.whitelist,
            'credibility_score': self._calculate_initial_credibility(domain),
            'last_updated': datetime.now().isoformat()
        }
        
        self.cache[url] = results
        return results

    def _calculate_initial_credibility(self, domain: str) -> float:
        score = 0.5
        if domain in self.whitelist:
            score += 0.3
        if domain in self.blacklist:
            score -= 0.3
        return max(0.0, min(1.0, score))