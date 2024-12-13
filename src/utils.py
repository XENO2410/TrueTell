# src/utils.py
import nltk
import os
import textblob
from textblob.download_corpora import download_lite
import subprocess
import sys

def download_nltk_data():
    """Download required NLTK and TextBlob data"""
    try:
        # Create a directory for NLTK data if it doesn't exist
        nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)

        # Download required NLTK data
        resources = [
            'punkt',
            'stopwords',
            'averaged_perceptron_tagger',
            'maxent_ne_chunker',
            'words',
            'punkt_tab',  # Added this
            'brown',      # Added this
            'wordnet'     # Added this
        ]

        for resource in resources:
            try:
                nltk.download(resource, quiet=True, raise_on_error=True)
            except Exception as e:
                print(f"Error downloading {resource}: {e}")

        # Download TextBlob corpora using subprocess
        try:
            print("Downloading TextBlob corpora...")
            subprocess.check_call([sys.executable, "-m", "textblob.download_corpora"])
            print("TextBlob corpora downloaded successfully")
        except Exception as e:
            print(f"Error downloading TextBlob corpora: {e}")
            # Fallback to download_lite
            try:
                download_lite()
            except Exception as e2:
                print(f"Error in fallback TextBlob download: {e2}")

    except Exception as e:
        print(f"Error setting up NLTK/TextBlob data: {e}")