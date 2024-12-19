# src/broadcast/analyzer.py
from typing import Dict, List, Optional
import asyncio
from datetime import datetime
from collections import deque
import numpy as np
from .stream import BroadcastMessage
from knowledge_graph import KnowledgeGraph

class BroadcastAnalyzer:
    def __init__(self, detector, knowledge_graph: KnowledgeGraph):
        self.detector = detector
        self.knowledge_graph = knowledge_graph
        self.analysis_buffer = deque(maxlen=1000)
        self.analysis_stats = {
            'total_processed': 0,
            'high_risk_count': 0,
            'avg_risk_score': 0.0,
            'start_time': datetime.now().isoformat()
        }

    async def process_message(self, message: BroadcastMessage) -> Dict:
        """Process a broadcast message in real-time"""
        try:
            # Validate message
            if not message.text or not message.timestamp:
                print("Invalid message format")
                return None
                
            # Analyze content using detector
            analysis_results = self.detector.analyze_text(message.text)
            if not analysis_results:
                return None

            # Get first result (since analyze_text returns a list)
            base_analysis = analysis_results[0]
                
            # Add broadcast-specific analysis
            broadcast_analysis = {
                'temporal_context': self._analyze_temporal_context(message),
                'source_reliability': self._check_source_reliability(message.source),
                'narrative_patterns': self._detect_narrative_patterns(message.text),
                'live_impact_score': self._calculate_live_impact(message)
            }
            
            # Combine analyses
            final_analysis = {
                **base_analysis,
                'broadcast_analysis': broadcast_analysis,
                'timestamp': message.timestamp
            }
            
            # Update knowledge graph
            try:
                self.knowledge_graph.add_content({
                    'text': message.text,
                    'timestamp': message.timestamp,
                    'source': message.source,
                    'analysis': final_analysis
                })
            except Exception as e:
                print(f"Knowledge graph update error: {e}")
            
            # Update statistics
            self._update_stats(final_analysis)
            
            # Store in buffer
            self.analysis_buffer.append({
                'message': message,
                'analysis': final_analysis,
                'timestamp': datetime.now().isoformat()
            })
            
            return final_analysis
            
        except Exception as e:
            print(f"Error processing broadcast message: {e}")
            return None

    def _analyze_temporal_context(self, message: BroadcastMessage) -> Dict:
        """Analyze temporal context of the message"""
        recent_messages = list(self.analysis_buffer)[-10:]
        
        return {
            'message_frequency': len(recent_messages) / 10 if recent_messages else 0,
            'similar_messages': self._find_similar_messages(message, recent_messages),
            'pattern_detection': self._detect_temporal_patterns(recent_messages)
        }

    def _check_source_reliability(self, source: str) -> Dict:
        """Check reliability of the broadcast source"""
        return {
            'source': source,
            'reliability_score': 0.75,  # Default good reliability
            'previous_violations': [],
            'verification_status': 'verified'
        }

    def _calculate_live_impact(self, message: BroadcastMessage) -> float:
        """Calculate potential impact of live broadcast content"""
        # Simple impact scoring
        return 0.5

    def _find_similar_messages(self, message: BroadcastMessage, recent_messages: List[Dict]) -> List[Dict]:
        """Find similar messages in recent history"""
        similar_messages = []
        for recent in recent_messages:
            if recent['message'].text == message.text:
                continue
            try:
                # Simple similarity based on common words
                current_words = set(message.text.lower().split())
                recent_words = set(recent['message'].text.lower().split())
                similarity = len(current_words & recent_words) / len(current_words | recent_words)
                if similarity > 0.5:
                    similar_messages.append({
                        'text': recent['message'].text,
                        'similarity': similarity,
                        'timestamp': recent['timestamp']
                    })
            except Exception as e:
                print(f"Error computing similarity: {e}")
        return similar_messages

    def _detect_temporal_patterns(self, recent_messages: List[Dict]) -> Dict:
        """Detect patterns in message timing"""
        if not recent_messages:
            return {'frequency': 0, 'patterns': []}
        
        timestamps = [datetime.fromisoformat(m['timestamp']) for m in recent_messages]
        time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                      for i in range(1, len(timestamps))]
        
        return {
            'frequency': len(timestamps) / (max(time_diffs) if time_diffs else 1),
            'patterns': []
        }

    def _find_repeated_phrases(self, text: str, recent_texts: List[str]) -> List[str]:
        """Find commonly repeated phrases"""
        return []  # Implement if needed

    def _detect_narrative_patterns(self, text: str) -> Dict:
        """Detect narrative patterns in broadcast"""
        recent_texts = [item['message'].text for item in self.analysis_buffer]
        
        return {
            'repeated_phrases': self._find_repeated_phrases(text, recent_texts),
            'narrative_shift': self._detect_narrative_shift(text, recent_texts),
            'topic_evolution': self._analyze_topic_evolution(text, recent_texts)
        }

    def _detect_narrative_shift(self, text: str, recent_texts: List[str]) -> Dict:
        """Detect shifts in narrative"""
        return {'shifts': [], 'confidence': 0.0}

    def _analyze_topic_evolution(self, text: str, recent_texts: List[str]) -> Dict:
        """Analyze how topics evolve over time"""
        return {'topics': [], 'evolution': []}

    def _update_stats(self, analysis_result: Dict):
        """Update analysis statistics"""
        self.analysis_stats['total_processed'] += 1
        
        if analysis_result.get('risk_score', 0) > 0.7:
            self.analysis_stats['high_risk_count'] += 1
            
        # Update running average
        current_avg = self.analysis_stats['avg_risk_score']
        n = self.analysis_stats['total_processed']
        new_score = analysis_result.get('risk_score', 0)
        self.analysis_stats['avg_risk_score'] = (current_avg * (n-1) + new_score) / n

    def get_current_stats(self) -> Dict:
        """Get current analysis statistics"""
        return {
            **self.analysis_stats,
            'buffer_size': len(self.analysis_buffer),
            'recent_risk_scores': [
                item['analysis'].get('risk_score', 0) 
                for item in list(self.analysis_buffer)[-10:]
            ]
        }