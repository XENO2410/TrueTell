# source_checker.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import whois
from datetime import datetime
import ssl
import socket

class SourceChecker:
    def __init__(self):
        self.cache = {}  # Cache results to avoid repeated checks
        self.blacklist = self._load_blacklist()
        self.whitelist = self._load_whitelist()

    def _load_blacklist(self):
        # Load known unreliable sources
        # This could be expanded to load from a database
        return set([
            'fake-news.com',
            'conspiracy-daily.com',
            # Add more
        ])

    def _load_whitelist(self):
        # Load known reliable sources
        return set([
            'reuters.com',
            'apnews.com',
            'bbc.com',
            # Add more
        ])

    async def check_source(self, url: str) -> Dict:
        """Comprehensive source credibility check"""
        if url in self.cache:
            return self.cache[url]

        domain = urlparse(url).netloc
        
        results = {
            'domain': domain,
            'website_age': await self._check_domain_age(domain),
            'ssl_valid': self._check_ssl(domain),
            'blacklisted': domain in self.blacklist,
            'whitelisted': domain in self.whitelist,
            'metrics': await self._get_site_metrics(url),
            'last_updated': datetime.now().isoformat()
        }

        # Calculate overall credibility score
        results['credibility_score'] = self._calculate_credibility_score(results)
        
        self.cache[url] = results
        return results

    async def _check_domain_age(self, domain: str) -> Optional[int]:
        """Check domain age in years"""
        try:
            w = whois.whois(domain)
            if w.creation_date:
                if isinstance(w.creation_date, list):
                    creation_date = w.creation_date[0]
                else:
                    creation_date = w.creation_date
                age = (datetime.now() - creation_date).days / 365
                return age
        except Exception:
            return None
        return None

    def _check_ssl(self, domain: str) -> bool:
        """Check if domain has valid SSL certificate"""
        try:
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443)) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    return True
        except Exception:
            return False

    async def _get_site_metrics(self, url: str) -> Dict:
        """Get various metrics about the website"""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            return {
                'has_about_page': bool(soup.find('a', href=lambda x: x and 'about' in x.lower())),
                'has_contact_info': bool(soup.find('a', href=lambda x: x and 'contact' in x.lower())),
                'has_privacy_policy': bool(soup.find('a', href=lambda x: x and 'privacy' in x.lower())),
                'article_length': len(response.text),
                'has_author': bool(soup.find(['author', 'byline'])),
                'has_dates': bool(soup.find(class_=lambda x: x and 'date' in x.lower()))
            }
        except Exception:
            return {}

    def _calculate_credibility_score(self, results: Dict) -> float:
        """Calculate overall credibility score"""
        score = 0.5  # Base score
        
        if results['whitelisted']:
            score += 0.3
        if results['blacklisted']:
            score -= 0.3
        if results['ssl_valid']:
            score += 0.1
            
        # Domain age factor
        if results['website_age']:
            if results['website_age'] > 5:
                score += 0.1
            elif results['website_age'] > 2:
                score += 0.05
                
        # Site metrics factors
        metrics = results['metrics']
        for metric in ['has_about_page', 'has_contact_info', 'has_privacy_policy', 'has_author', 'has_dates']:
            if metrics.get(metric):
                score += 0.02
                
        return max(0.0, min(1.0, score))