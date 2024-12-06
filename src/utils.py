# src/utils.py
import nltk
import os

def download_nltk_data():
    """Download required NLTK data"""
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
            'words'
        ]

        for resource in resources:
            try:
                nltk.download(resource, quiet=True, raise_on_error=True)
            except Exception as e:
                print(f"Error downloading {resource}: {e}")

    except Exception as e:
        print(f"Error setting up NLTK data: {e}")