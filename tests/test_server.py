# tests/test_server.py
import asyncio
import websockets
import json
from datetime import datetime
import random

async def broadcast_feed(websocket, path):
    """Simulate broadcast feed"""
    test_messages = [
        "Breaking: Major technological breakthrough announced in renewable energy.",
        "ALERT: Unverified reports claim health issues! #conspiracy",
        "Scientists discover new species in the Amazon rainforest.",
        "URGENT: Government documents reveal shocking details! Must share!",
        "Latest update: Market manipulation by algorithms, experts warn.",
        "BREAKING: Revolutionary cure discovered! Limited time offer!",
        "New study confirms environmental impacts on weather patterns.",
        "EXPOSED: Hidden operations controlling media! Share now!"
    ]
    
    try:
        while True:
            message = {
                'text': random.choice(test_messages),
                'timestamp': datetime.now().isoformat(),
                'source': 'test_broadcast',
                'metadata': {
                    'type': 'news',
                    'priority': random.choice(['high', 'medium', 'low'])
                }
            }
            await websocket.send(json.dumps(message))
            await asyncio.sleep(2)
            
    except websockets.exceptions.ConnectionClosed:
        pass

async def main():
    async with websockets.serve(broadcast_feed, "localhost", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())