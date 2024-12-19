# src/broadcast/stream.py
from dataclasses import dataclass
from typing import Dict, List, Callable, Optional
from datetime import datetime
import websockets
import json
import asyncio
from .sources import NewsSource
from .social_monitor import SocialMediaMonitor
import os

@dataclass
class BroadcastMessage:
    text: str
    timestamp: str
    source: str
    metadata: Dict

class BroadcastStream:
    def __init__(self):
        self.connections = {}
        self.listeners: List[Callable] = []
        self.is_running = False
        self.buffer_size = 1000
        self.message_buffer = []
        
        # Initialize with better error handling
        news_api_key = os.getenv('NEWS_API_KEY')
        twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        
        if news_api_key:
            self.news_source = NewsSource(news_api_key)
        else:
            print("Warning: NEWS_API_KEY not found in environment variables")
            self.news_source = None
            
        if twitter_bearer_token:
            self.social_monitor = SocialMediaMonitor({
                'bearer_token': twitter_bearer_token
            })
        else:
            print("Warning: TWITTER_BEARER_TOKEN not found in environment variables")
            self.social_monitor = None

    async def fetch_all_sources(self) -> List[Dict]:
        """Fetch from all available sources"""
        all_items = []
        
        try:
            if self.news_source:
                news_items = await self.news_source.fetch_news()
                all_items.extend(news_items)
                
            if self.social_monitor:
                social_items = await self.social_monitor.monitor_twitter()
                all_items.extend(social_items)
                
        except Exception as e:
            print(f"Error fetching from sources: {e}")
            
        return all_items
        
    async def connect(self, stream_url: str) -> bool:
        """Connect to a broadcast stream"""
        try:
            self.websocket = await websockets.connect(stream_url)
            self.is_running = True
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    async def start_streaming(self):
        """Start processing the broadcast stream"""
        if not self.is_running:
            return
            
        try:
            while self.is_running:
                message = await self.websocket.recv()
                parsed_message = self._parse_message(message)
                if parsed_message:
                    await self._process_message(parsed_message)
        except Exception as e:
            print(f"Streaming error: {e}")
            self.is_running = False

    def add_listener(self, callback: Callable):
        """Add a listener for broadcast messages"""
        self.listeners.append(callback)

    async def _process_message(self, message: BroadcastMessage):
        """Process incoming broadcast message"""
        # Add to buffer
        self.message_buffer.append(message)
        if len(self.message_buffer) > self.buffer_size:
            self.message_buffer.pop(0)

        # Notify listeners
        for listener in self.listeners:
            try:
                await listener(message)
            except Exception as e:
                print(f"Listener error: {e}")

    def _parse_message(self, raw_message: str) -> Optional[BroadcastMessage]:
        """Parse incoming message into BroadcastMessage"""
        try:
            data = json.loads(raw_message)
            return BroadcastMessage(
                text=data.get('text', ''),
                timestamp=data.get('timestamp', datetime.now().isoformat()),
                source=data.get('source', 'unknown'),
                metadata=data.get('metadata', {})
            )
        except Exception as e:
            print(f"Message parsing error: {e}")
            return None

    def get_buffer_stats(self) -> Dict:
        """Get statistics about the message buffer"""
        return {
            'buffer_size': len(self.message_buffer),
            'oldest_message': self.message_buffer[0].timestamp if self.message_buffer else None,
            'newest_message': self.message_buffer[-1].timestamp if self.message_buffer else None
        }

    async def stop(self):
        """Stop the broadcast stream"""
        self.is_running = False
        if hasattr(self, 'websocket'):
            await self.websocket.close()