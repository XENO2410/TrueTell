# src/integration_layer.py
from typing import Dict, List, Any
import requests
import json
from datetime import datetime
import asyncio
from abc import ABC, abstractmethod
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class IntegrationProvider(ABC):
    @abstractmethod
    async def fetch_data(self) -> Dict:
        pass
    
    @abstractmethod
    async def send_alert(self, alert_data: Dict) -> bool:
        pass
    
    @abstractmethod
    def send_alert_sync(self, alert_data: Dict) -> bool:
        pass

class APIIntegration(IntegrationProvider):
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def fetch_data(self) -> Dict:
        try:
            response = requests.get(
                f"{self.base_url}/data",
                headers=self.headers
            )
            return response.json()
        except Exception as e:
            print(f"Error fetching data: {e}")
            return {}
    
    async def send_alert(self, alert_data: Dict) -> bool:
        try:
            response = requests.post(
                f"{self.base_url}/alerts",
                headers=self.headers,
                json=alert_data
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending alert: {e}")
            return False
            
    def send_alert_sync(self, alert_data: Dict) -> bool:
        try:
            response = requests.post(
                f"{self.base_url}/alerts",
                headers=self.headers,
                json=alert_data
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending alert: {e}")
            return False

class IntegrationLayer:
    def __init__(self):
        self.providers: Dict[str, IntegrationProvider] = {}
        self.data_cache: Dict[str, Any] = {}
        self.alert_queue: List[Dict] = []
    
    def register_provider(self, name: str, provider: IntegrationProvider):
        """Register a new integration provider"""
        self.providers[name] = provider
    
    async def fetch_all_data(self) -> Dict[str, Any]:
        """Fetch data from all registered providers"""
        results = {}
        for name, provider in self.providers.items():
            results[name] = await provider.fetch_data()
        return results
    
    async def send_alerts(self, alert_data: Dict) -> Dict[str, bool]:
        """Send alerts to all registered providers"""
        results = {}
        for name, provider in self.providers.items():
            results[name] = await provider.send_alert(alert_data)
        return results
    
    def send_alerts_sync(self, alert_data: Dict) -> Dict[str, bool]:
        """Synchronous method to send alerts to all registered providers"""
        results = {}
        for name, provider in self.providers.items():
            try:
                # Try synchronous send first
                if hasattr(provider, 'send_alert_sync'):
                    results[name] = provider.send_alert_sync(alert_data)
                else:
                    # Fallback to async if no sync method
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results[name] = loop.run_until_complete(provider.send_alert(alert_data))
                    loop.close()
            except Exception as e:
                print(f"Error sending alert via {name}: {e}")
                results[name] = False
        return results
    
    def cache_data(self, key: str, data: Any):
        """Cache data with timestamp"""
        self.data_cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    def get_cached_data(self, key: str) -> Any:
        """Retrieve cached data"""
        return self.data_cache.get(key, {}).get('data')
    
    async def process_alert_queue(self):
        """Process queued alerts"""
        while self.alert_queue:
            alert = self.alert_queue.pop(0)
            await self.send_alerts(alert)
    
    def queue_alert(self, alert_data: Dict):
        """Add alert to queue"""
        self.alert_queue.append(alert_data)

class WebhookIntegration(IntegrationProvider):
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    async def fetch_data(self) -> Dict:
        return {}  # Webhooks don't fetch data
    
    async def send_alert(self, alert_data: Dict) -> bool:
        try:
            response = requests.post(
                self.webhook_url,
                json=alert_data
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending webhook: {e}")
            return False
            
    def send_alert_sync(self, alert_data: Dict) -> bool:
        try:
            response = requests.post(
                self.webhook_url,
                json=alert_data
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending webhook: {e}")
            return False

class SlackIntegration(IntegrationProvider):
    def __init__(self, slack_token: str, channel: str):
        self.slack_token = slack_token
        self.channel = channel
        self.headers = {
            "Authorization": f"Bearer {slack_token}",
            "Content-Type": "application/json"
        }
    
    async def fetch_data(self) -> Dict:
        return {}  # Slack integration doesn't fetch data
    
    async def send_alert(self, alert_data: Dict) -> bool:
        try:
            message = {
                "channel": self.channel,
                "text": f"🚨 Alert: {alert_data['message']}\nRisk Score: {alert_data['risk_score']:.2%}\nTimestamp: {alert_data['timestamp']}"
            }
            response = requests.post(
                "https://slack.com/api/chat.postMessage",
                headers=self.headers,
                json=message
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending Slack message: {e}")
            return False
            
    def send_alert_sync(self, alert_data: Dict) -> bool:
        try:
            message = {
                "channel": self.channel,
                "text": f"🚨 Alert: {alert_data['message']}\nRisk Score: {alert_data['risk_score']:.2%}\nTimestamp: {alert_data['timestamp']}"
            }
            response = requests.post(
                "https://slack.com/api/chat.postMessage",
                headers=self.headers,
                json=message
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending Slack message: {e}")
            return False

class EmailIntegration(IntegrationProvider):
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password

    async def fetch_data(self) -> Dict:
        return {}  # Email integration doesn't fetch data

    async def send_alert(self, alert_data: Dict) -> bool:
        return self.send_alert_sync(alert_data)  # Reuse sync implementation

    def send_alert_sync(self, alert_data: Dict) -> bool:
        try:
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = alert_data.get('recipient', '')
            msg['Subject'] = f"Alert: {alert_data.get('subject', 'Security Alert')}"

            body = f"""
            Alert Details:
            Message: {alert_data.get('message', '')}
            Risk Score: {alert_data.get('risk_score', 0):.2%}
            Timestamp: {alert_data.get('timestamp', '')}
            """

            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            return True
        except Exception as e:
            print(f"Error sending email: {e}")
            return False