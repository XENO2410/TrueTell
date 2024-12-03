# src/knowledge_graph.py
from neo4j import GraphDatabase
from config import Config

class KnowledgeGraph:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
        )
        
    def add_fact(self, statement, classification, sources):
        with self.driver.session() as session:
            session.write_transaction(
                self._create_fact_node,
                statement,
                classification,
                sources
            )
    
    @staticmethod
    def _create_fact_node(tx, statement, classification, sources):
        query = (
            "CREATE (f:Fact {statement: $statement, "
            "classification: $classification, "
            "sources: $sources})"
        )
        tx.run(query, statement=statement,
               classification=classification,
               sources=sources)