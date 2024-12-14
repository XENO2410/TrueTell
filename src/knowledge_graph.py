#src/knowledge_graph.py
from typing import Dict, List, Set, Tuple
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from collections import defaultdict
import spacy
from networkx.readwrite import json_graph
import streamlit as st
import numpy as np
import plotly.express as px
import networkx as nx
import re
import pandas as pd


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        self.entity_types = {
            'PERSON', 'ORG', 'GPE', 'EVENT', 'DATE', 
            'TOPIC', 'CLAIM', 'SOURCE'
        }
        
        self.relationships = defaultdict(int)
        
        # Enhanced color scheme
        self.colors = {
            'CLAIM': '#1f77b4',    # blue
            'PERSON': '#2ca02c',   # green
            'ORG': '#ff7f0e',      # orange
            'SOURCE': '#d62728',   # red
            'GPE': '#9467bd',      # purple
            'EVENT': '#8c564b',    # brown
            'DATE': '#e377c2',     # pink
            'TOPIC': '#7f7f7f'     # gray
        }
        
        # Add visualization settings
        self.viz_settings = {
            'node_size': 20,
            'edge_width': 0.5,
            'show_labels': True,
            'layout': 'spring'
        }

    def _extract_timestamp_from_text(self, text: str) -> datetime:
        """Extract timestamp from text with various formats"""
        try:
            # Common timestamp patterns
            patterns = [
                r'\((\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))\)',  # (2:30 PM)
                r'(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))',      # 2:30 PM
                r'\((\d{2}:\d{2})\)',                       # (14:30)
                r'(\d{2}:\d{2})'                            # 14:30
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    time_str = match.group(1)
                    try:
                        # Try 12-hour format first
                        return datetime.strptime(
                            f"{datetime.now().strftime('%Y-%m-%d')} {time_str}", 
                            "%Y-%m-%d %I:%M %p"
                        )
                    except ValueError:
                        try:
                            # Try 24-hour format
                            return datetime.strptime(
                                f"{datetime.now().strftime('%Y-%m-%d')} {time_str}", 
                                "%Y-%m-%d %H:%M"
                            )
                        except ValueError:
                            continue
            
            # If no timestamp found, use current time
            return datetime.now()
        except Exception as e:
            print(f"Error extracting timestamp: {e}")
            return datetime.now()
    
    def add_content(self, content: Dict) -> None:
        """Add new content to the knowledge graph"""
        try:
            # Process text with SpaCy
            doc = self.nlp(content['text'])
            
            # Extract timestamp from text if not provided
            if 'timestamp' not in content or not content['timestamp']:
                content['timestamp'] = self._extract_timestamp_from_text(content['text']).strftime('%Y-%m-%d %H:%M:%S')
            
            # Extract entities
            entities = self._extract_entities(doc)
            
            # Add claim node
            claim_id = f"claim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.graph.add_node(claim_id, 
                              type='CLAIM',
                              text=content['text'],
                              timestamp=content['timestamp'],
                              credibility_score=content.get('credibility_score', 0.5),
                              verified=content.get('verified', False))
            
            # Add entities and their relationships
            for entity, entity_type in entities:
                entity_id = f"{entity_type}_{entity}".lower()
                
                # Add entity node if it doesn't exist
                if not self.graph.has_node(entity_id):
                    self.graph.add_node(entity_id,
                                      type=entity_type,
                                      name=entity,
                                      first_seen=datetime.now().isoformat())
                
                # Connect entity to claim
                self.graph.add_edge(claim_id, entity_id, 
                                  relationship='MENTIONS',
                                  timestamp=content['timestamp'])  # Use extracted timestamp
                
                # Update relationship counter
                self.relationships[f'CLAIM-{entity_type}'] += 1
            
            # Add source if available
            if 'source' in content:
                source_id = f"SOURCE_{content['source']}".lower()
                self.graph.add_node(source_id,
                                  type='SOURCE',
                                  name=content['source'],
                                  reliability_score=content.get('reliability_score', 0.5))
                self.graph.add_edge(claim_id, source_id,
                                  relationship='FROM_SOURCE',
                                  timestamp=content['timestamp'])  # Use extracted timestamp
            
            # Add related claims based on entity overlap
            self._link_related_claims(claim_id, entities)
            
        except Exception as e:
            print(f"Error adding content to knowledge graph: {e}")

    def _extract_entities(self, doc) -> List[Tuple[str, str]]:
        """Extract entities from text using SpaCy"""
        entities = []
        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                entities.append((ent.text, ent.label_))
        return entities

    def _link_related_claims(self, claim_id: str, entities: List[Tuple[str, str]]) -> None:
        """Link claims that share entities"""
        entity_ids = [f"{entity_type}_{entity}".lower() for entity, entity_type in entities]
        
        # Find other claims that mention these entities
        for node in self.graph.nodes():
            if node.startswith('claim_') and node != claim_id:
                shared_entities = set(self.graph.neighbors(node)) & set(entity_ids)
                if shared_entities:
                    self.graph.add_edge(claim_id, node,
                                      relationship='RELATED_TO',
                                      shared_entities=list(shared_entities),
                                      timestamp=datetime.now().isoformat())

    def get_statistics(self) -> Dict:
        """Get statistics about the knowledge graph with debug info"""
        try:
            stats = {
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges(),
                'node_types': defaultdict(int)
            }
            
            # Count nodes by type
            for _, attr in self.graph.nodes(data=True):
                node_type = attr.get('type', 'unknown')
                stats['node_types'][node_type] += 1
            
            # Convert defaultdict to regular dict
            stats['node_types'] = dict(stats['node_types'])
            
            # Add relationship statistics
            stats['relationships'] = dict(self.relationships)
            
            # Add debug information
            stats['debug'] = {
                'nodes': list(self.graph.nodes(data=True)),
                'edges': list(self.graph.edges(data=True))
            }
            
            print(f"Knowledge Graph Stats: {stats}")  # Debug line
            return stats
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            # Return default values if there's an error
            return {
                'total_nodes': 0,
                'total_edges': 0,
                'node_types': {},
                'relationships': {},
                'debug': {
                    'error': str(e),
                    'nodes': [],
                    'edges': []
                }
            }

    def visualize(self, min_weight=0.1, show_labels=True, layout='spring', node_size=20, highlight_nodes=None, container=None):
        """
        Visualize the knowledge graph using Plotly with enhanced options
        
        Parameters:
        -----------
        min_weight : float
            Minimum weight threshold for displaying relationships
        show_labels : bool
            Whether to show node labels
        layout : str
            Type of layout ('spring', 'circular', 'random')
        node_size : int
            Size of nodes in visualization
        highlight_nodes : list
            List of node IDs to highlight
        container : streamlit.container
            Streamlit container to render the visualization
        """
        try:
            # Create position layout based on selected type
            if layout == 'circular':
                pos = nx.circular_layout(self.graph)
            elif layout == 'random':
                pos = nx.random_layout(self.graph)
            else:  # default to spring layout
                pos = nx.spring_layout(self.graph)
            
            # Prepare node traces for each type
            node_traces = {}
            for node_type in self.colors:
                nodes = [n for n, attr in self.graph.nodes(data=True) 
                        if attr.get('type') == node_type]
                if nodes:
                    x_pos = [pos[node][0] for node in nodes]
                    y_pos = [pos[node][1] for node in nodes]
                    
                    # Prepare node text and size
                    node_text = []
                    node_sizes = []
                    for node in nodes:
                        text = f"{self.graph.nodes[node].get('type', 'Unknown')}"
                        if show_labels:
                            text += f"<br>{self.graph.nodes[node].get('name', node)}"
                        node_text.append(text)
                        
                        # Adjust size if node is highlighted
                        if highlight_nodes and node in highlight_nodes:
                            node_sizes.append(node_size * 1.5)
                        else:
                            node_sizes.append(node_size)
                    
                    node_traces[node_type] = go.Scatter(
                        x=x_pos,
                        y=y_pos,
                        mode='markers+text' if show_labels else 'markers',
                        name=node_type,
                        marker=dict(
                            size=node_sizes,
                            color=self.colors[node_type],
                            line=dict(width=2, color='white')
                        ),
                        text=node_text,
                        textposition="top center",
                        hoverinfo='text',
                        showlegend=True,
                        textfont=dict(color='white')
                    )
    
            # Prepare edge trace with weight filtering
            edge_x = []
            edge_y = []
            edge_text = []
            
            for edge in self.graph.edges(data=True):
                weight = edge[2].get('weight', 1.0)
                if weight >= min_weight:
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_text.append(edge[2].get('relationship', 'connects to'))
    
            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='text',
                mode='lines',
                text=edge_text,
                showlegend=False
            )
    
            # Create figure with dark theme
            fig = go.Figure(
                data=[edge_trace] + list(node_traces.values()),
                layout=go.Layout(
                    title={
                        'text': 'Knowledge Graph Visualization',
                        'font': {'color': 'white'}
                    },
                    showlegend=True,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(
                        showgrid=False, 
                        zeroline=False, 
                        showticklabels=False,
                        showline=False
                    ),
                    yaxis=dict(
                        showgrid=False, 
                        zeroline=False, 
                        showticklabels=False,
                        showline=False
                    ),
                    plot_bgcolor='rgb(17, 17, 17)',
                    paper_bgcolor='rgb(17, 17, 17)',
                    legend={
                        'font': {'color': 'white'},
                        'bgcolor': 'rgba(0,0,0,0)'
                    }
                )
            )
    
            # Display in Streamlit
            if container:
                container.plotly_chart(fig, use_container_width=True)
            else:
                st.plotly_chart(fig, use_container_width=True)
    
            # Add statistics below visualization
            stats = self.get_statistics()
            if container:
                col1, col2, col3 = container.columns(3)
            else:
                col1, col2, col3 = st.columns(3)
    
            with col1:
                st.metric("Total Nodes", stats['total_nodes'])
            with col2:
                st.metric("Total Relationships", stats['total_edges'])
            with col3:
                st.metric("Entity Types", len(stats['node_types']))
    
            return True
    
        except Exception as e:
            if container:
                container.error(f"Error in visualization: {e}")
            else:
                st.error(f"Error in visualization: {e}")
            return False

    def export_graph(self, filepath: str) -> None:
        """Export the graph to JSON format"""
        data = json_graph.node_link_data(self.graph)
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def import_graph(self, filepath: str) -> None:
        """Import the graph from JSON format"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.graph = json_graph.node_link_graph(data)

    def get_entity_relationships(self, entity_id: str) -> Dict:
        """Get all relationships for an entity"""
        if not self.graph.has_node(entity_id):
            return {}
        
        relationships = {
            'mentions': [],
            'sources': [],
            'related_entities': []
        }
        
        for neighbor in self.graph.neighbors(entity_id):
            edge_data = self.graph.get_edge_data(entity_id, neighbor)
            if neighbor.startswith('claim_'):
                relationships['mentions'].append({
                    'claim_id': neighbor,
                    'timestamp': edge_data['timestamp']
                })
            elif neighbor.startswith('source_'):
                relationships['sources'].append({
                    'source_id': neighbor,
                    'timestamp': edge_data['timestamp']
                })
            else:
                relationships['related_entities'].append({
                    'entity_id': neighbor,
                    'relationship': edge_data['relationship']
                })
        
        return relationships

    def get_claim_context(self, claim_id: str) -> Dict:
        """Get contextual information for a claim"""
        try:
            if not self.graph.has_node(claim_id):
                return {
                    'claim': {'text': 'Claim not found', 'type': 'CLAIM'},
                    'entities': [],
                    'sources': [],
                    'related_claims': []
                }
            
            context = {
                'claim': self.graph.nodes[claim_id],
                'entities': [],
                'sources': [],
                'related_claims': []
            }
            
            # Ensure claim has required fields
            if 'text' not in context['claim']:
                context['claim']['text'] = 'No text available'
            if 'type' not in context['claim']:
                context['claim']['type'] = 'CLAIM'
            
            for neighbor in self.graph.neighbors(claim_id):
                try:
                    node_data = self.graph.nodes[neighbor]
                    edge_data = self.graph.get_edge_data(claim_id, neighbor)
                    
                    if node_data.get('type') == 'SOURCE':
                        context['sources'].append({
                            'name': node_data.get('name', 'Unknown source'),
                            'reliability_score': node_data.get('reliability_score', 0.5)
                        })
                    elif neighbor.startswith('claim_'):
                        context['related_claims'].append({
                            'claim_id': neighbor,
                            'text': node_data.get('text', 'No text available'),
                            'shared_entities': edge_data.get('shared_entities', [])
                        })
                    else:
                        context['entities'].append({
                            'name': node_data.get('name', 'Unknown entity'),
                            'type': node_data.get('type', 'Unknown type')
                        })
                except Exception as e:
                    print(f"Error processing neighbor {neighbor}: {e}")
                    continue
            
            return context
            
        except Exception as e:
            print(f"Error getting claim context: {e}")
            return {
                'claim': {'text': 'Error retrieving claim', 'type': 'CLAIM'},
                'entities': [],
                'sources': [],
                'related_claims': []
            }
            
    def analyze_claim_patterns(self, analysis_type: str = "Entity Co-occurrence", 
                             min_occurrence: int = 2,
                             entity_types: List[str] = None,
                             time_window: str = "Day",
                             include_weekends: bool = True,
                             chain_length: int = 3,
                             similarity_threshold: float = 0.7) -> Dict:
        """
        Analyze patterns in claims with enhanced misinformation detection
        """
        patterns = {
            'common_entities': defaultdict(int),
            'source_reliability': defaultdict(list),
            'temporal_patterns': defaultdict(int),
            'narrative_chains': [],
            'co_occurrence_network': None,
            'temporal_heatmap': None,
            'temporal_metadata': {
                'days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                        'Friday', 'Saturday', 'Sunday'],
                'hours': list(range(24)),
                'max_value': 0,
                'min_value': 0
            },
            'misinformation_indicators': {
                'high_risk_claims': [],
                'suspicious_entities': defaultdict(float),
                'verification_status': defaultdict(int),
                'credibility_distribution': defaultdict(int),
                'source_credibility': defaultdict(list)
            },
            'analysis_metadata': {
                'total_claims': 0,
                'verified_claims': 0,
                'avg_credibility': 0.0,
                'risk_level': 'LOW'
            }
        }
    
        try:
            # Get all claim nodes and initialize metadata
            claim_nodes = [n for n in self.graph.nodes() if n.startswith('claim_')]
            patterns['analysis_metadata']['total_claims'] = len(claim_nodes)
            credibility_scores = []
    
            # Process all claims first for metadata
            for claim in claim_nodes:
                attrs = self.graph.nodes[claim]
                credibility = attrs.get('credibility_score', 0.5)
                credibility_scores.append(credibility)
                
                # Track verification status
                verified = attrs.get('verified', False)
                patterns['misinformation_indicators']['verification_status'][str(verified)] += 1
                if verified:
                    patterns['analysis_metadata']['verified_claims'] += 1
                
                # Track high-risk claims
                if credibility < 0.3:  # High risk threshold
                    patterns['misinformation_indicators']['high_risk_claims'].append({
                        'claim_id': claim,
                        'text': attrs.get('text', ''),
                        'credibility': credibility,
                        'timestamp': attrs.get('timestamp', ''),
                        'entities': list(self.graph.neighbors(claim))
                    })
    
                # Track credibility distribution
                credibility_bin = round(credibility * 10) / 10  # Round to nearest 0.1
                patterns['misinformation_indicators']['credibility_distribution'][str(credibility_bin)] += 1
    
            # Calculate average credibility and set risk level
            if credibility_scores:
                avg_cred = sum(credibility_scores) / len(credibility_scores)
                patterns['analysis_metadata']['avg_credibility'] = avg_cred
                patterns['analysis_metadata']['risk_level'] = (
                    'HIGH' if avg_cred < 0.3 else 'MEDIUM' if avg_cred < 0.7 else 'LOW'
                )
    
            if analysis_type == "Entity Co-occurrence":
                entity_pairs = defaultdict(int)
                entity_credibility = defaultdict(list)
                
                for claim in claim_nodes:
                    entities = [n for n in self.graph.neighbors(claim)
                              if not n.startswith('claim_')]
                    
                    if entity_types:
                        entities = [e for e in entities 
                                  if self.graph.nodes[e].get('type') in entity_types]
                    
                    # Track entity credibility
                    claim_credibility = self.graph.nodes[claim].get('credibility_score', 0.5)
                    for entity in entities:
                        patterns['common_entities'][entity] += 1
                        entity_credibility[entity].append(claim_credibility)
                    
                    # Track co-occurrences with credibility impact
                    for i, e1 in enumerate(entities):
                        for e2 in entities[i+1:]:
                            pair = tuple(sorted([e1, e2]))
                            entity_pairs[pair] += 1
                
                # Calculate suspicious entities
                for entity, credibilities in entity_credibility.items():
                    avg_cred = sum(credibilities) / len(credibilities)
                    if avg_cred < 0.3:  # High risk threshold
                        patterns['misinformation_indicators']['suspicious_entities'][entity] = avg_cred
                
                # Create enhanced co-occurrence network
                G = nx.Graph()
                for (e1, e2), weight in entity_pairs.items():
                    if weight >= min_occurrence:
                        # Calculate edge risk based on entity credibility
                        e1_cred = sum(entity_credibility[e1]) / len(entity_credibility[e1])
                        e2_cred = sum(entity_credibility[e2]) / len(entity_credibility[e2])
                        edge_risk = (2 - (e1_cred + e2_cred)) / 2  # Higher value = higher risk
                        
                        G.add_edge(e1, e2, 
                                  weight=weight,
                                  risk_score=edge_risk,
                                  credibility_score=(e1_cred + e2_cred) / 2)
                
                patterns['co_occurrence_network'] = G
    
            elif analysis_type == "Temporal Patterns":
                # Initialize temporal analysis structures
                temporal_data = []
                heatmap_data = np.zeros((7, 24))
                risk_heatmap = np.zeros((7, 24))  # Track risk scores over time
                current_date = datetime.now()
                
                for node, attrs in self.graph.nodes(data=True):
                    if node.startswith('claim_'):
                        try:
                            text = attrs.get('text', '')
                            time_pattern = r'\((\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?\)'
                            time_match = re.search(time_pattern, text)
                            
                            if time_match:
                                hour = int(time_match.group(1))
                                minute = int(time_match.group(2))
                                ampm = time_match.group(3)
                                
                                if ampm and ampm.upper() == 'PM' and hour != 12:
                                    hour += 12
                                elif ampm and ampm.upper() == 'AM' and hour == 12:
                                    hour = 0
                                
                                timestamp = current_date.replace(
                                    hour=hour,
                                    minute=minute,
                                    second=0,
                                    microsecond=0
                                )
                                
                                temporal_data.append(timestamp)
                                
                                # Update both regular and risk heatmaps
                                day_idx = timestamp.weekday()
                                hour_idx = timestamp.hour
                                heatmap_data[day_idx][hour_idx] += 1
                                
                                # Add risk score to risk heatmap
                                risk_score = 1 - attrs.get('credibility_score', 0.5)
                                risk_heatmap[day_idx][hour_idx] += risk_score
                                
                        except Exception as e:
                            print(f"Error processing timestamp for node {node}: {e}")
                            continue
                
                if temporal_data:
                    temporal_data.sort()
                    time_windows = {
                        "Hour": timedelta(hours=1),
                        "Day": timedelta(days=1),
                        "Week": timedelta(weeks=1),
                        "Month": timedelta(days=30)
                    }
                    window = time_windows.get(time_window, timedelta(days=1))
                    
                    if len(temporal_data) > 0:
                        min_time = min(temporal_data)
                        max_time = max(temporal_data)
                        current = min_time
                        
                        while current <= max_time:
                            next_window = current + window
                            window_claims = [t for t in temporal_data if current <= t < next_window]
                            patterns['temporal_patterns'][current.strftime('%Y-%m-%d %H:%M:%S')] = len(window_claims)
                            current = next_window
                        
                        # Update metadata with risk information
                        patterns['temporal_metadata'].update({
                            'max_value': float(np.max(heatmap_data)),
                            'min_value': float(np.min(heatmap_data)),
                            'total_entries': len(temporal_data),
                            'average_per_slot': float(np.mean(heatmap_data)),
                            'max_risk_hour': int(np.argmax(np.sum(risk_heatmap, axis=0))),
                            'max_risk_day': int(np.argmax(np.sum(risk_heatmap, axis=1))),
                            'risk_heatmap': risk_heatmap.tolist(),
                            'time_range': {
                                'start': min_time.strftime('%I:%M %p'),
                                'end': max_time.strftime('%I:%M %p'),
                                'duration_minutes': int((max_time - min_time).total_seconds() / 60)
                            }
                        })
                        
                        patterns['temporal_heatmap'] = heatmap_data.tolist()
                        
                        # Enhanced distribution statistics
                        patterns['temporal_metadata']['distribution'] = {
                            'total_events': len(temporal_data),
                            'unique_hours': len(set(t.hour for t in temporal_data)),
                            'unique_days': len(set(t.weekday() for t in temporal_data)),
                            'events_by_hour': {f"{h:02d}:00": int(sum(heatmap_data[:, h])) 
                                             for h in range(24) if sum(heatmap_data[:, h]) > 0},
                            'risk_by_hour': {f"{h:02d}:00": float(sum(risk_heatmap[:, h]))
                                           for h in range(24) if sum(risk_heatmap[:, h]) > 0}
                        }
    
            else:  # Narrative Chains
                visited = set()
                
                for claim in claim_nodes:
                    if claim not in visited:
                        chain = []
                        queue = [(claim, [])]
                        
                        while queue:
                            current, path = queue.pop(0)
                            if current not in visited:
                                visited.add(current)
                                
                                if current.startswith('claim_'):
                                    attrs = self.graph.nodes[current]
                                    chain.append({
                                        'claim_id': current,
                                        'text': attrs.get('text', ''),
                                        'timestamp': attrs.get('timestamp', ''),
                                        'credibility_score': attrs.get('credibility_score', 0.5),
                                        'verified': attrs.get('verified', False),
                                        'entities': list(self.graph.neighbors(current))
                                    })
                                
                                for neighbor in self.graph.neighbors(current):
                                    if (neighbor.startswith('claim_') and 
                                        neighbor not in visited and 
                                        neighbor not in path):
                                        similarity = self._calculate_claim_similarity(current, neighbor)
                                        if similarity >= similarity_threshold:
                                            queue.append((neighbor, path + [current]))
                        
                        if len(chain) >= chain_length:
                            # Calculate chain risk metrics
                            chain_credibility = [c['credibility_score'] for c in chain]
                            avg_credibility = sum(chain_credibility) / len(chain_credibility)
                            credibility_trend = chain_credibility[-1] - chain_credibility[0]
                            verification_ratio = sum(1 for c in chain if c['verified']) / len(chain)
                            
                            # Create chain visualization with risk information
                            chain_graph = nx.DiGraph()
                            for i in range(len(chain)-1):
                                chain_graph.add_edge(
                                    chain[i]['claim_id'],
                                    chain[i+1]['claim_id'],
                                    credibility_delta=chain_credibility[i+1] - chain_credibility[i]
                                )
                            
                            chain.append({
                                'chain_graph': chain_graph,
                                'avg_credibility': avg_credibility,
                                'credibility_trend': credibility_trend,
                                'verification_ratio': verification_ratio,
                                'risk_level': 'HIGH' if avg_credibility < 0.3 else 'MEDIUM' if avg_credibility < 0.7 else 'LOW'
                            })
                            
                            patterns['narrative_chains'].append(chain)
            
            return patterns
            
        except Exception as e:
            st.error(f"Error in pattern analysis: {str(e)}")
            return patterns

    def visualize_temporal_patterns(self, patterns: Dict) -> None:
        """Visualize temporal patterns with enhanced graphics"""
        try:
            if patterns['temporal_patterns']:
                if 'time_range' in patterns['temporal_metadata']:
                    st.write(f"**Time Range:** {patterns['temporal_metadata']['time_range']['start']} - "
                            f"{patterns['temporal_metadata']['time_range']['end']} "
                            f"({patterns['temporal_metadata']['time_range']['duration_minutes']} minutes)")
                            
                # Convert to DataFrame for timeline
                df = pd.DataFrame([
                    {'timestamp': datetime.strptime(k, '%Y-%m-%d %H:%M:%S'), 'count': v}
                    for k, v in patterns['temporal_patterns'].items()
                ])
                
                # Sort by timestamp
                df = df.sort_values('timestamp')
                
                # Create timeline visualization
                fig_timeline = px.line(
                    df,
                    x='timestamp',
                    y='count',
                    title="Content Distribution Over Time",
                    template="plotly_dark"
                )
                
                fig_timeline.update_traces(line_color='#00ff00')
                fig_timeline.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Number of Events",
                    plot_bgcolor='rgb(17, 17, 17)',
                    paper_bgcolor='rgb(17, 17, 17)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig_timeline, use_container_width=True)
    
                # Heatmap visualization
                if patterns['temporal_heatmap']:
                    heatmap_data = np.array(patterns['temporal_heatmap'])
                    
                    # Create custom colorscale with better contrast
                    colorscale = [
                        [0, 'rgb(68, 1, 84)'],        # Dark purple for zero
                        [0.001, 'rgb(59, 82, 139)'],   # Blue-purple for very low
                        [0.25, 'rgb(33, 145, 140)'],   # Teal for low-mid
                        [0.5, 'rgb(94, 201, 98)'],     # Light green for mid
                        [0.75, 'rgb(253, 231, 37)'],   # Yellow for high
                        [1, 'rgb(255, 255, 255)']      # White for maximum
                    ]
                    
                    # Create heatmap
                    fig_heatmap = px.imshow(
                        heatmap_data,
                        labels=dict(x="Hour of Day", y="Day of Week"),
                        x=list(range(24)),
                        y=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                        title="Activity Distribution (Events per Hour)",
                        color_continuous_scale=colorscale,
                        aspect='auto'
                    )
                    
                    fig_heatmap.update_layout(
                        xaxis_title="Hour of Day",
                        yaxis_title="Day of Week",
                        coloraxis_colorbar_title="Activity Level",
                        plot_bgcolor='rgb(17, 17, 17)',
                        paper_bgcolor='rgb(17, 17, 17)',
                        font=dict(color='white')
                    )
                    
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    # Add statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Peak Activity",
                            f"{patterns['temporal_metadata']['max_value']:.0f} events/hour"
                        )
                    with col2:
                        st.metric(
                            "Average Activity",
                            f"{patterns['temporal_metadata']['average_per_slot']:.1f} events/hour"
                        )
                    with col3:
                        st.metric(
                            "Total Events",
                            patterns['temporal_metadata']['total_entries']
                        )
                    
                    # Add distribution information
                    if 'distribution' in patterns['temporal_metadata']:
                        st.write("Distribution Statistics:")
                        st.json(patterns['temporal_metadata']['distribution'])
            else:
                st.info("No temporal pattern data available.")
                
        except Exception as e:
            st.error(f"Error visualizing temporal patterns: {str(e)}")

    def visualize_misinformation_patterns(self, patterns: Dict) -> None:
        """Visualize misinformation patterns and indicators"""
        try:
            # Display overall statistics
            st.subheader("Misinformation Analysis Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                verified_ratio = (patterns['metadata']['verified_claims'] / 
                                patterns['metadata']['total_claims'] if patterns['metadata']['total_claims'] > 0 else 0)
                st.metric("Verification Rate", f"{verified_ratio:.1%}")
            
            with col2:
                st.metric("High Risk Claims", patterns['metadata']['high_risk_claims'])
            
            with col3:
                st.metric("Average Credibility", f"{patterns['metadata']['avg_credibility']:.2f}")
            
            with col4:
                st.metric("Total Claims", patterns['metadata']['total_claims'])
    
            # Display suspicious patterns
            if patterns['misinformation_indicators']['suspicious_patterns']:
                st.subheader("🚨 Suspicious Patterns Detected")
                for pattern in patterns['misinformation_indicators']['suspicious_patterns']:
                    with st.expander(f"Suspicious Claim: {pattern['text'][:100]}..."):
                        st.warning(f"Credibility Score: {pattern['credibility']:.2f}")
                        st.write("**Related Entities:**")
                        for entity in pattern['entities']:
                            entity_risk = patterns['misinformation_indicators']['entity_risk_scores'].get(entity, 0)
                            st.write(f"- {entity}: Risk Score {entity_risk:.2f}")
                        st.caption(f"Timestamp: {pattern['timestamp']}")
    
            # Visualize credibility distribution
            if patterns['misinformation_indicators']['credibility_scores']:
                st.subheader("Credibility Score Distribution")
                fig = px.histogram(
                    x=patterns['misinformation_indicators']['credibility_scores'],
                    nbins=20,
                    title="Distribution of Credibility Scores",
                    labels={'x': 'Credibility Score', 'y': 'Count'},
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
    
            # Display entity risk analysis
            if patterns['misinformation_indicators']['entity_risk_scores']:
                st.subheader("Entity Risk Analysis")
                
                # Create risk score DataFrame
                risk_df = pd.DataFrame([
                    {'Entity': entity, 'Risk Score': score}
                    for entity, score in patterns['misinformation_indicators']['entity_risk_scores'].items()
                ]).sort_values('Risk Score', ascending=False)
    
                # Show top risky entities
                fig = px.bar(
                    risk_df.head(10),
                    x='Entity',
                    y='Risk Score',
                    title="Top 10 Entities by Risk Score",
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
    
            # Display narrative chain analysis
            if patterns['narrative_chains']:
                st.subheader("Narrative Chain Analysis")
                for idx, chain in enumerate(patterns['narrative_chains']):
                    if isinstance(chain[-1], dict) and 'risk_score' in chain[-1]:
                        risk_score = chain[-1]['risk_score']
                        verification_ratio = chain[-1]['verification_ratio']
                        
                        with st.expander(f"Narrative Chain {idx+1} (Risk: {risk_score:.2f})"):
                            st.progress(1 - risk_score)  # Inverse of risk score
                            st.write(f"Verification Ratio: {verification_ratio:.1%}")
                            
                            for claim in chain[:-1]:  # Exclude the metadata dict
                                st.write(f"- {claim['text']}")
                                st.caption(f"Credibility: {claim['credibility_score']:.2f}")
    
        except Exception as e:
            st.error(f"Error visualizing misinformation patterns: {str(e)}")
              
    def _calculate_claim_similarity(self, claim1: str, claim2: str) -> float:
        """Calculate similarity between two claims"""
        try:
            # Get claim texts
            text1 = self.graph.nodes[claim1].get('text', '')
            text2 = self.graph.nodes[claim2].get('text', '')
            
            # Use SpaCy for similarity
            doc1 = self.nlp(text1)
            doc2 = self.nlp(text2)
            
            return doc1.similarity(doc2)
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def _find_narrative_chains(self) -> List[Dict]:
        """Find chains of related claims that might form narratives"""
        chains = []
        visited = set()
        
        try:
            for node in self.graph.nodes():
                if node.startswith('claim_') and node not in visited:
                    chain = self._explore_chain(node, visited)
                    if len(chain) > 1:  # Only include chains with multiple claims
                        chains.append(chain)
            return chains
        except Exception as e:
            print(f"Error finding narrative chains: {e}")
            return []
    
    def _explore_chain(self, start_node: str, visited: Set[str]) -> List[Dict]:
        """Explore a chain of related claims"""
        chain = []
        queue = [(start_node, [])]
        
        while queue:
            node, path = queue.pop(0)
            if node not in visited:
                visited.add(node)
                
                # Add node to chain
                if node.startswith('claim_'):
                    chain.append({
                        'claim_id': node,
                        'text': self.graph.nodes[node].get('text', ''),
                        'timestamp': self.graph.nodes[node].get('timestamp', ''),
                        'credibility_score': self.graph.nodes[node].get('credibility_score', 0.5)
                    })
                
                # Explore neighbors
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited and neighbor.startswith('claim_'):
                        queue.append((neighbor, path + [node]))
        
        return chain

    def get_entities_by_type(self, types: List[str]) -> List[Dict]:
        """Get entities filtered by type"""
        entities = []
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get('type') in types:
                relationships = self.get_entity_relationships(node)
                entities.append({
                    'id': node,
                    'name': attrs.get('name', node),
                    'type': attrs.get('type'),
                    'relationships': relationships
                })
        return entities
    
    def search_entities(self, query: str, search_type: str = "Contains", entity_types: List[str] = None, relationship_types: List[str] = None) -> List[Dict]:
        """
        Search entities with enhanced filtering options
        
        Parameters:
        -----------
        query : str
            Search query string
        search_type : str
            Type of search ("Contains", "Exact Match", "Regex")
        entity_types : List[str]
            List of entity types to filter by
        relationship_types : List[str]
            List of relationship types to filter by
        """
        try:
            results = []
            query = query.lower() if search_type != "Regex" else query
            
            for node, attrs in self.graph.nodes(data=True):
                # Skip if entity type doesn't match filter
                if entity_types and attrs.get('type') not in entity_types:
                    continue
                    
                name = attrs.get('name', '').lower()
                text = attrs.get('text', '').lower()
                
                # Apply search based on search_type
                match = False
                if search_type == "Contains":
                    match = query in name or query in text
                elif search_type == "Exact Match":
                    match = query == name or query == text
                elif search_type == "Regex":
                    import re
                    try:
                        pattern = re.compile(query, re.IGNORECASE)
                        match = pattern.search(name) is not None or pattern.search(text) is not None
                    except re.error:
                        continue
                
                if match:
                    # Get relationships for the entity
                    relationships = self.get_entity_relationships(node)
                    
                    # Filter by relationship types if specified
                    if relationship_types:
                        filtered_relationships = {}
                        for rel_type, rel_data in relationships.items():
                            if rel_type in relationship_types:
                                filtered_relationships[rel_type] = rel_data
                        relationships = filtered_relationships
                    
                    # Add to results if it has matching relationships or no relationship filter
                    if not relationship_types or relationships:
                        results.append({
                            'id': node,
                            'name': attrs.get('name', node),
                            'type': attrs.get('type'),
                            'text': attrs.get('text', ''),
                            'relationships': relationships,
                            'first_seen': attrs.get('first_seen', ''),
                            'mention_count': len(relationships.get('mentions', [])),
                            'source_count': len(relationships.get('sources', []))
                        })
            
            return results
            
        except Exception as e:
            print(f"Error in search_entities: {e}")
            return []
    
    def sort_entities(self, entities: List[Dict], sort_by: str = "Name") -> List[Dict]:
        """Sort entities based on specified criteria"""
        try:
            if sort_by == "Name":
                return sorted(entities, key=lambda x: x['name'].lower())
            elif sort_by == "Type":
                return sorted(entities, key=lambda x: x['type'])
            elif sort_by == "Relationship Count":
                return sorted(entities, 
                            key=lambda x: len(x['relationships'].get('mentions', [])) + 
                                        len(x['relationships'].get('sources', [])), 
                            reverse=True)
            elif sort_by == "Mention Count":
                return sorted(entities, 
                            key=lambda x: len(x['relationships'].get('mentions', [])), 
                            reverse=True)
            return entities
        except Exception as e:
            print(f"Error sorting entities: {e}")
            return entities

    def visualize_network(self, network: nx.Graph, title: str = "Entity Co-occurrence Network") -> None:
        """
        Visualize a network graph using Plotly
        
        Parameters:
        -----------
        network : networkx.Graph
            Network to visualize
        title : str
            Title for the visualization
        """
        try:
            if not network or network.number_of_nodes() == 0:
                st.info("No network data to visualize.")
                return
            
            # Create layout
            pos = nx.spring_layout(network)
            
            # Create edge trace
            edge_x = []
            edge_y = []
            edge_weights = []
            
            for edge in network.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                weight = edge[2].get('weight', 1)
                edge_weights.extend([weight, weight, None])
            
            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(
                    width=1,
                    color='rgba(150, 150, 150, 0.5)'
                ),
                hoverinfo='none',
                mode='lines'
            )
            
            # Create node traces by type
            node_traces = {}
            
            for node in network.nodes():
                # Get node type and attributes
                node_type = self.graph.nodes[node].get('type', 'Unknown')
                node_color = self.colors.get(node_type, '#888')
                
                if node_type not in node_traces:
                    node_traces[node_type] = {
                        'x': [],
                        'y': [],
                        'text': [],
                        'names': [],
                        'sizes': []
                    }
                
                # Add node position
                x, y = pos[node]
                node_traces[node_type]['x'].append(x)
                node_traces[node_type]['y'].append(y)
                
                # Add node text and name
                node_name = self.graph.nodes[node].get('name', node)
                node_traces[node_type]['names'].append(node_name)
                
                # Calculate node size based on degree
                size = 20 + 10 * network.degree(node)
                node_traces[node_type]['sizes'].append(size)
                
                # Create hover text
                hover_text = (
                    f"Type: {node_type}<br>"
                    f"Name: {node_name}<br>"
                    f"Connections: {network.degree(node)}"
                )
                node_traces[node_type]['text'].append(hover_text)
            
            # Create Plotly figure
            fig = go.Figure(
                data=[edge_trace],
                layout=go.Layout(
                    title=title,
                    showlegend=True,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='rgb(17, 17, 17)',
                    paper_bgcolor='rgb(17, 17, 17)',
                    font=dict(color='white')
                )
            )
            
            # Add node traces to figure
            for node_type, trace_data in node_traces.items():
                node_trace = go.Scatter(
                    x=trace_data['x'],
                    y=trace_data['y'],
                    mode='markers',
                    name=node_type,
                    marker=dict(
                        color=self.colors.get(node_type, '#888'),
                        size=trace_data['sizes'],
                        line=dict(width=2, color='white')
                    ),
                    text=trace_data['text'],
                    hoverinfo='text'
                )
                fig.add_trace(node_trace)
            
            # Add network statistics
            stats_text = (
                f"Nodes: {network.number_of_nodes()}<br>"
                f"Edges: {network.number_of_edges()}<br>"
                f"Avg. Degree: {sum(dict(network.degree()).values())/network.number_of_nodes():.2f}"
            )
            
            fig.add_annotation(
                text=stats_text,
                xref="paper",
                yref="paper",
                x=0,
                y=1.1,
                showarrow=False,
                font=dict(color='white'),
                align="left"
            )
            
            # Update layout for better visibility
            fig.update_layout(
                legend=dict(
                    x=1.1,
                    y=0.5,
                    bgcolor='rgba(0,0,0,0)',
                    bordercolor='rgba(255,255,255,0.3)',
                    font=dict(color='white')
                )
            )
            
            # Display in Streamlit
            st.plotly_chart(fig, use_container_width=True)
            
            # Add network metrics
            with st.expander("Network Metrics"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    density = nx.density(network)
                    st.metric("Network Density", f"{density:.3f}")
                
                with col2:
                    if nx.is_connected(network):
                        avg_path = nx.average_shortest_path_length(network)
                        st.metric("Avg. Path Length", f"{avg_path:.2f}")
                    else:
                        st.metric("Avg. Path Length", "N/A (Disconnected)")
                
                with col3:
                    clustering_coef = nx.average_clustering(network)
                    st.metric("Clustering Coefficient", f"{clustering_coef:.3f}")
            
        except Exception as e:
            st.error(f"Error visualizing network: {str(e)}")
        

    def visualize_entity_relationships(self, entity_name: str) -> None:
        """Visualize relationships for a specific entity"""
        try:
            # Create subgraph for entity and its neighbors
            entity_nodes = [entity_name] + list(self.graph.neighbors(entity_name))
            subgraph = self.graph.subgraph(entity_nodes)
            
            # Create position layout
            pos = nx.spring_layout(subgraph)
            
            # Create figure with entity-specific visualization
            node_traces = []
            edge_traces = []
            
            # Add nodes
            for node in subgraph.nodes():
                node_type = subgraph.nodes[node].get('type', 'Unknown')
                color = self.colors.get(node_type, '#7f7f7f')
                
                node_trace = go.Scatter(
                    x=[pos[node][0]],
                    y=[pos[node][1]],
                    mode='markers+text',
                    name=node_type,
                    marker=dict(
                        size=30 if node == entity_name else 20,
                        color=color,
                        line=dict(width=2, color='white')
                    ),
                    text=[f"{node_type}<br>{subgraph.nodes[node].get('name', node)}"],
                    textposition="top center",
                    hoverinfo='text'
                )
                node_traces.append(node_trace)
            
            # Add edges
            edge_x = []
            edge_y = []
            edge_text = []
            for edge in subgraph.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_text.append(edge[2].get('relationship', 'connects to'))
            
            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='text',
                mode='lines',
                text=edge_text
            )
            
            # Create figure
            fig = go.Figure(
                data=[edge_trace] + node_traces,
                layout=go.Layout(
                    title=f"Relationships for {entity_name}",
                    showlegend=True,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    plot_bgcolor='rgb(17, 17, 17)',
                    paper_bgcolor='rgb(17, 17, 17)',
                    font=dict(color='white')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error visualizing entity relationships: {e}")
    
    def export_as_image(self, filename: str) -> bool:
        """Export graph visualization as PNG image"""
        try:
            fig = self._create_figure()
            fig.write_image(filename)
            return True
        except Exception as e:
            print(f"Error exporting as image: {e}")
            return False
    
    def export_as_html(self, filename: str) -> bool:
        """Export graph visualization as interactive HTML"""
        try:
            fig = self._create_figure()
            fig.write_html(filename)
            return True
        except Exception as e:
            print(f"Error exporting as HTML: {e}")
            return False
    
    def _create_figure(self) -> go.Figure:
        """Create a Plotly figure for the graph"""
        pos = nx.spring_layout(self.graph)
        
        # Create traces
        node_traces = {}
        for node_type in self.colors:
            nodes = [n for n, attr in self.graph.nodes(data=True) 
                    if attr.get('type') == node_type]
            if nodes:
                x_pos = [pos[node][0] for node in nodes]
                y_pos = [pos[node][1] for node in nodes]
                
                node_text = [f"{self.graph.nodes[node].get('type', 'Unknown')}<br>"
                            f"{self.graph.nodes[node].get('name', node)}"
                            for node in nodes]
                
                node_traces[node_type] = go.Scatter(
                    x=x_pos,
                    y=y_pos,
                    mode='markers+text',
                    name=node_type,
                    marker=dict(
                        size=20,
                        color=self.colors[node_type],
                        line=dict(width=2, color='white')
                    ),
                    text=node_text,
                    textposition="top center",
                    hoverinfo='text'
                )
    
        # Create edge trace
        edge_x = []
        edge_y = []
        edge_text = []
        for edge in self.graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(edge[2].get('relationship', 'connects to'))
    
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='text',
            mode='lines',
            text=edge_text
        )
    
        # Create figure
        fig = go.Figure(
            data=[edge_trace] + list(node_traces.values()),
            layout=go.Layout(
                title='Knowledge Graph Visualization',
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                plot_bgcolor='rgb(17, 17, 17)',
                paper_bgcolor='rgb(17, 17, 17)',
                font=dict(color='white')
            )
        )
        
        return fig
    
    def search_nodes(self, query: str) -> List[str]:
        """Search for nodes matching the query"""
        matching_nodes = []
        query = query.lower()
        
        for node, attrs in self.graph.nodes(data=True):
            name = attrs.get('name', '').lower()
            text = attrs.get('text', '').lower()
            if query in name or query in text:
                matching_nodes.append(node)
        
        return matching_nodes
    
    @property
    def relationship_types(self) -> Set[str]:
        """Get all relationship types in the graph"""
        types = set()
        for _, _, data in self.graph.edges(data=True):
            if 'relationship' in data:
                types.add(data['relationship'])
        return types