# src\broadcast\sources.py
import requests
from typing import List, Dict
from datetime import datetime, timedelta
import os
from newsapi import NewsApiClient
import asyncio

class NewsSource:
    def __init__(self, api_key: str):
        self.newsapi = NewsApiClient(api_key=api_key)
        self.last_fetch_time = None
        self.processed_ids = set()  # Track processed articles
        
    async def fetch_news(self) -> List[Dict]:
        """Fetch news from NewsAPI"""
        try:
            current_time = datetime.now()
            
            # Rate limiting
            if self.last_fetch_time and (current_time - self.last_fetch_time).seconds < 60:
                return []
                
            self.last_fetch_time = current_time
            
            # Get news
            response = self.newsapi.get_everything(
                q='news',  # Broader search term
                language='en',
                sort_by='publishedAt',
                page_size=10
            )
            
            # Process and filter new articles
            articles = []
            for article in response['articles']:
                article_id = hash(article['url'])
                if article_id not in self.processed_ids:
                    self.processed_ids.add(article_id)
                    articles.append({
                        'id': article_id,
                        'title': article['title'],
                        'description': article['description'],
                        'text': f"{article['title']}. {article['description'] or ''}",
                        'timestamp': article['publishedAt'],
                        'source': article['source']['name'],
                        'url': article['url'],
                        'author': article['author']
                    })
            
            return articles
            
        except Exception as e:
            print(f"NewsAPI error: {e}")
            return []