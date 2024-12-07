# src/fact_database.py
from typing import Dict, List, Optional
import sqlite3
import json
from datetime import datetime
import pandas as pd

class FactDatabase:
    def __init__(self, db_path: str = "facts.db"):
        self.db_path = db_path
        self.initialize_database()

    def initialize_database(self):
        """Initialize the SQLite database with necessary tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create facts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            claim TEXT NOT NULL,
            rating TEXT NOT NULL,
            source TEXT,
            source_url TEXT,
            verification_date TIMESTAMP,
            credibility_score FLOAT,
            category TEXT,
            explanation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Create sources table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            reliability_score FLOAT,
            domain TEXT,
            category TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Create fact_references table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS fact_references (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact_id INTEGER,
            reference_url TEXT,
            reference_title TEXT,
            reference_date TIMESTAMP,
            FOREIGN KEY (fact_id) REFERENCES facts (id)
        )
        ''')

        conn.commit()
        conn.close()

    def add_fact(self, claim: str, rating: str, source: str, 
                 credibility_score: float, category: str = None, 
                 explanation: str = None, references: List[Dict] = None):
        """Add a new fact to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
            INSERT INTO facts (claim, rating, source, credibility_score, 
                             category, explanation, verification_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (claim, rating, source, credibility_score, category, 
                 explanation, datetime.now()))

            fact_id = cursor.lastrowid

            # Add references if provided
            if references:
                for ref in references:
                    cursor.execute('''
                    INSERT INTO fact_references (fact_id, reference_url, 
                                               reference_title, reference_date)
                    VALUES (?, ?, ?, ?)
                    ''', (fact_id, ref.get('url'), ref.get('title'), 
                         ref.get('date')))

            conn.commit()
            return fact_id
        except Exception as e:
            print(f"Error adding fact: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()

    def search_facts(self, claim: str, threshold: float = 0.7) -> List[Dict]:
        """Search for similar facts in the database"""
        conn = sqlite3.connect(self.db_path)
        try:
            # Using pandas for better text matching
            df = pd.read_sql_query("SELECT * FROM facts", conn)
            
            # Calculate similarity scores (basic implementation)
            df['similarity'] = df['claim'].apply(
                lambda x: self._calculate_similarity(x, claim)
            )
            
            # Filter by threshold
            matches = df[df['similarity'] >= threshold].to_dict('records')
            
            return [{
                'claim': match['claim'],
                'rating': match['rating'],
                'source': match['source'],
                'credibility_score': match['credibility_score'],
                'category': match['category'],
                'explanation': match['explanation'],
                'similarity': match['similarity']
            } for match in matches]
        finally:
            conn.close()

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (basic implementation)"""
        # Convert to sets of words
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0

    def get_source_reliability(self, source: str) -> float:
        """Get the reliability score for a source"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                "SELECT reliability_score FROM sources WHERE name = ?", 
                (source,)
            )
            result = cursor.fetchone()
            return result[0] if result else 0.5
        finally:
            conn.close()

    def export_database(self, format: str = 'json') -> str:
        """Export the database content"""
        conn = sqlite3.connect(self.db_path)
        
        if format == 'json':
            facts_df = pd.read_sql_query("SELECT * FROM facts", conn)
            sources_df = pd.read_sql_query("SELECT * FROM sources", conn)
            refs_df = pd.read_sql_query("SELECT * FROM fact_references", conn)
            
            export_data = {
                'facts': facts_df.to_dict('records'),
                'sources': sources_df.to_dict('records'),
                'references': refs_df.to_dict('records')
            }
            
            return json.dumps(export_data, default=str)
        
        conn.close()