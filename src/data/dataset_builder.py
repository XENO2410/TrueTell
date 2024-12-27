# src/data/dataset_builder.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import os
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

class IndianDatasetBuilder:
    def __init__(self, save_dir: str = "datasets"):
        """Initialize the dataset builder"""
        self.save_dir = save_dir
        self.sources = {
            'fact_check_sites': [
                'https://factly.in',
                'https://www.altnews.in',
                'https://www.boomlive.in',
                'https://www.factchecker.in'
            ],
            'news_sites': [
                'https://timesofindia.indiatimes.com',
                'https://www.ndtv.com',
                'https://www.thehindu.com'
            ]
        }
        self.setup_logging()
        os.makedirs(save_dir, exist_ok=True)

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('dataset_builder.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DatasetBuilder')

    async def collect_fact_checks(self) -> List[Dict]:
        """Collect fact-checking articles"""
        fact_checks = []
        
        for site in self.sources['fact_check_sites']:
            try:
                articles = await self._scrape_site(site)
                fact_checks.extend(articles)
                self.logger.info(f"Collected {len(articles)} articles from {site}")
            except Exception as e:
                self.logger.error(f"Error collecting from {site}: {e}")
        
        return fact_checks

    async def collect_news_articles(self) -> List[Dict]:
        """Collect news articles for verification"""
        articles = []
        
        for site in self.sources['news_sites']:
            try:
                site_articles = await self._scrape_site(site)
                articles.extend(site_articles)
                self.logger.info(f"Collected {len(site_articles)} articles from {site}")
            except Exception as e:
                self.logger.error(f"Error collecting from {site}: {e}")
        
        return articles

    def create_training_dataset(self, 
                              fact_checks: List[Dict], 
                              news_articles: List[Dict]) -> pd.DataFrame:
        """Create labeled dataset for training"""
        dataset = []
        
        # Process fact-checks
        for article in fact_checks:
            try:
                processed = self._process_article(article, is_fact_check=True)
                if processed:
                    dataset.append(processed)
            except Exception as e:
                self.logger.error(f"Error processing fact-check: {e}")

        # Process news articles
        for article in news_articles:
            try:
                processed = self._process_article(article, is_fact_check=False)
                if processed:
                    dataset.append(processed)
            except Exception as e:
                self.logger.error(f"Error processing news article: {e}")

        # Convert to DataFrame
        df = pd.DataFrame(dataset)
        
        # Add regional language translations
        df = self._add_translations(df)
        
        # Save dataset
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df.to_csv(f"{self.save_dir}/indian_misinfo_dataset_{timestamp}.csv", index=False)
        
        return df

    def _process_article(self, article: Dict, is_fact_check: bool) -> Dict:
        """Process and label individual articles"""
        return {
            'text': article.get('text', ''),
            'title': article.get('title', ''),
            'source': article.get('source', ''),
            'date': article.get('date', ''),
            'language': self._detect_language(article.get('text', '')),
            'is_fact_check': is_fact_check,
            'credibility_score': self._calculate_credibility(article),
            'entities': self._extract_indian_entities(article.get('text', '')),
            'regional_context': self._extract_regional_context(article),
            'verification_status': article.get('verification_status', 'unverified')
        }

    def _add_translations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add translations for major Indian languages"""
        try:
            from googletrans import Translator
            translator = Translator()
            
            languages = ['hi', 'bn', 'te', 'ta', 'mr']  # Hindi, Bengali, Telugu, Tamil, Marathi
            
            for lang in languages:
                df[f'text_{lang}'] = df['text'].apply(
                    lambda x: translator.translate(x, dest=lang).text if pd.notnull(x) else ''
                )
                
            return df
        except Exception as e:
            self.logger.error(f"Error in translation: {e}")
            return df

    def _extract_indian_entities(self, text: str) -> List[str]:
        """Extract India-specific named entities"""
        # Add custom entity recognition for Indian names, places, etc.
        pass

    def _extract_regional_context(self, article: Dict) -> Dict:
        """Extract region-specific context"""
        # Add regional context extraction
        pass

    def _calculate_credibility(self, article: Dict) -> float:
        """Calculate credibility score"""
        # Add credibility scoring logic
        pass

    def _detect_language(self, text: str) -> str:
        """Detect text language"""
        # Add language detection logic
        pass

    async def _scrape_site(self, url: str) -> List[Dict]:
        """Scrape articles from a site"""
        # Add web scraping logic
        pass