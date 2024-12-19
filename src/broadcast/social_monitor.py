# src\broadcast\social_monitor.py
import tweepy
from typing import List, Dict
from datetime import datetime, timedelta
import time

class SocialMediaMonitor:
    def __init__(self, auth_config: Dict):
        self.client = tweepy.Client(
            bearer_token=auth_config['bearer_token'],
            wait_on_rate_limit=True
        )
        self.last_fetch_time = datetime.now()
        self.rate_limit_window = 900  # 15 minutes in seconds
        self.max_requests = 450  # Twitter API v2 rate limit
        self.request_count = 0

    async def monitor_twitter(self) -> List[Dict]:
        """Monitor Twitter for relevant content"""
        try:
            # Check rate limiting
            current_time = datetime.now()
            time_elapsed = (current_time - self.last_fetch_time).total_seconds()
            
            if time_elapsed >= self.rate_limit_window:
                # Reset counters if rate limit window has passed
                self.last_fetch_time = current_time
                self.request_count = 0
            elif self.request_count >= self.max_requests:
                print("Rate limit reached, waiting...")
                return []

            # Increment request counter
            self.request_count += 1

            # Search query
            query = 'misinformation OR "fake news" OR disinformation -is:retweet'
            
            # Get tweets
            tweets = self.client.search_recent_tweets(
                query=query,
                tweet_fields=['created_at', 'public_metrics'],
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
                    'id': tweet.id
                }
            } for tweet in tweets.data]

        except Exception as e:
            print(f"Twitter monitoring error: {e}")
            return []