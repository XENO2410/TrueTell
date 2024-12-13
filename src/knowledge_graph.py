#src/knowledge_graph.py
from typing import Dict, List, Tuple, Set
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
import json
from collections import defaultdict
import spacy
from networkx.readwrite import json_graph
import streamlit as st

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        # Load SpaCy model for entity recognition
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
        
        # Color scheme for visualization
        self.colors = {
            'CLAIM': '#1f77b4',  # blue
            'PERSON': '#2ca02c',  # green
            'ORG': '#ff7f0e',    # orange
            'SOURCE': '#d62728',  # red
            'GPE': '#9467bd',    # purple
            'EVENT': '#8c564b',   # brown
            'DATE': '#e377c2',    # pink
            'TOPIC': '#7f7f7f'    # gray
        }

    def add_content(self, content: Dict) -> None:
        """Add new content to the knowledge graph"""
        try:
            # Process text with SpaCy
            doc = self.nlp(content['text'])
            
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
                                  timestamp=datetime.now().isoformat())
                
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
                                  timestamp=datetime.now().isoformat())
            
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

    def visualize(self, container=None, unique_id=None):
        """
        Visualize the knowledge graph using Plotly
        Args:
            container: Optional Streamlit container to render the visualization
            unique_id: Unique identifier for the visualization instance
        """
        try:
            # Generate unique ID if not provided
            if unique_id is None:
                unique_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    
            # Create position layout
            pos = nx.spring_layout(self.graph)
            
            # Prepare node traces for each type
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
                        hoverinfo='text',
                        showlegend=True
                    )
    
            # Prepare edge trace
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
                text=edge_text,
                showlegend=False
            )
    
            # Create figure
            fig = go.Figure(
                data=[edge_trace] + list(node_traces.values()),
                layout=go.Layout(
                    title='Knowledge Graph Visualization',
                    showlegend=True,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='white'
                )
            )
    
            # Display in Streamlit with unique keys
            if container:
                container.plotly_chart(
                    fig, 
                    use_container_width=True, 
                    key=f"kg_plot_{unique_id}"
                )
            else:
                st.plotly_chart(
                    fig, 
                    use_container_width=True, 
                    key=f"kg_plot_{unique_id}"
                )
    
            # Add statistics below visualization with unique keys
            stats = self.get_statistics()
            if container:
                col1, col2, col3 = container.columns(3)
            else:
                col1, col2, col3 = st.columns(3)
    
            with col1:
                st.metric(
                    "Total Nodes", 
                    stats['total_nodes'],
                    key=f"total_nodes_{unique_id}"
                )
            with col2:
                st.metric(
                    "Total Relationships", 
                    stats['total_edges'],
                    key=f"total_edges_{unique_id}"
                )
            with col3:
                st.metric(
                    "Entity Types", 
                    len(stats['node_types']),
                    key=f"entity_types_{unique_id}"
                )
    
            # Add export button with unique key
            if container:
                if container.button(
                    "Export Graph", 
                    key=f"export_button_{unique_id}"
                ):
                    self.export_graph(f"knowledge_graph_{unique_id}.json")
                    container.success("Graph exported successfully!")
    
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