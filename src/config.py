# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')
    API_KEY = os.getenv('API_KEY', '')
    
    CONFIDENCE_THRESHOLD = 0.75
    MODEL_NAME = "facebook/bart-large-mnli"