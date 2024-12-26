# src/broadcast/social_monitor.py

import tweepy
import praw
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
import time
import os
from dotenv import load_dotenv

class SocialMediaMonitor:
    def __init__(self, auth_config: Dict = None):
        """Initialize social media monitor with authentication"""
        # Load environment variables
        load_dotenv()
        
        # Store auth config
        self.auth_config = auth_config or {}
        
        # Initialize API clients
        self.twitter_client = self._init_twitter()
        self.reddit_client = self._init_reddit()
        
        # Rate limiting settings
        self.last_fetch_time = datetime.now()
        self.rate_limit_window = 900  # 15 minutes
        self.max_requests = 450  # Twitter API v2 limit
        self.request_count = 0

    def _init_twitter(self) -> Optional[tweepy.Client]:
        """Initialize Twitter client with error handling"""
        try:
            # Get credentials from auth_config or environment
            bearer_token = (self.auth_config.get('bearer_token') or 
                          os.getenv('TWITTER_BEARER_TOKEN'))
            api_key = os.getenv('TWITTER_API_KEY')
            api_secret = os.getenv('TWITTER_API_SECRET')
            access_token = os.getenv('TWITTER_ACCESS_TOKEN')
            access_secret = os.getenv('TWITTER_ACCESS_SECRET')

            if not all([bearer_token, api_key, api_secret, access_token, access_secret]):
                print("Warning: Missing Twitter credentials")
                return None

            return tweepy.Client(
                bearer_token=bearer_token,
                consumer_key=api_key,
                consumer_secret=api_secret,
                access_token=access_token,
                access_token_secret=access_secret,
                wait_on_rate_limit=True
            )
        except Exception as e:
            print(f"Twitter initialization error: {e}")
            return None

    def _init_reddit(self) -> Optional[praw.Reddit]:
        """Initialize Reddit client with error handling"""
        try:
            client_id = os.getenv('REDDIT_CLIENT_ID')
            client_secret = os.getenv('REDDIT_CLIENT_SECRET')

            if not all([client_id, client_secret]):
                print("Warning: Missing Reddit credentials")
                return None

            return praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent="TruthTell/1.0"
            )
        except Exception as e:
            print(f"Reddit initialization error: {e}")
            return None

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        current_time = datetime.now()
        time_elapsed = (current_time - self.last_fetch_time).total_seconds()
        
        if time_elapsed >= self.rate_limit_window:
            self.last_fetch_time = current_time
            self.request_count = 0
            return True
            
        return self.request_count < self.max_requests

    async def monitor_twitter(self) -> List[Dict]:
        """Monitor Twitter for relevant content"""
        if not self.twitter_client:
            return []

        try:
            if not self._check_rate_limit():
                print("Rate limit reached, waiting...")
                return []

            self.request_count += 1
            
            # Search query
            query = 'misinformation OR "fake news" OR disinformation -is:retweet'
            
            # Get tweets
            tweets = self.twitter_client.search_recent_tweets(
                query=query,
                tweet_fields=['created_at', 'public_metrics', 'source'],
                max_results=10
            )

            if not tweets.data:
                return []

            # Format response
            return [{
                'text': tweet.text,
                'timestamp': tweet.created_at.isoformat(),
                'source': 'Twitter',
                'metadata': {
                    'metrics': tweet.public_metrics,
                    'source': tweet.source,
                    'id': tweet.id
                }
            } for tweet in tweets.data]

        except Exception as e:
            print(f"Twitter monitoring error: {e}")
            return []

    async def monitor_reddit(self) -> List[Dict]:
        """Monitor Reddit for relevant content"""
        if not self.reddit_client:
            return []

        try:
            subreddits = ['news', 'worldnews', 'politics']
            posts = []

            for subreddit in subreddits:
                try:
                    subreddit_posts = self.reddit_client.subreddit(subreddit).new(limit=10)
                    posts.extend([{
                        'text': post.title + '\n' + (post.selftext if post.selftext else ''),
                        'timestamp': datetime.fromtimestamp(post.created_utc).isoformat(),
                        'source': f'Reddit/r/{subreddit}',
                        'metadata': {
                            'score': post.score,
                            'url': post.url,
                            'id': post.id
                        }
                    } for post in subreddit_posts])
                except Exception as e:
                    print(f"Error fetching from r/{subreddit}: {e}")
                    continue

            return posts

        except Exception as e:
            print(f"Reddit monitoring error: {e}")
            return []

    async def monitor_all_platforms(self) -> List[Dict]:
        """Monitor all configured social media platforms"""
        tasks = [
            self.monitor_twitter(),
            self.monitor_reddit()
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out errors and combine results
            all_posts = []
            for result in results:
                if isinstance(result, list):
                    all_posts.extend(result)
                elif isinstance(result, Exception):
                    print(f"Platform monitoring error: {result}")
                    
            # Sort by timestamp
            return sorted(all_posts, key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            print(f"Error monitoring platforms: {e}")
            return []

    def get_status(self) -> Dict:
        """Get monitor status"""
        return {
            'twitter_enabled': self.twitter_client is not None,
            'reddit_enabled': self.reddit_client is not None,
            'rate_limit': {
                'requests_remaining': self.max_requests - self.request_count,
                'reset_time': self.last_fetch_time.isoformat(),
                'window_seconds': self.rate_limit_window
            }
        }