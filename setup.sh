#!/bin/bash

# Download NLTK data
python -m nltk.downloader punkt
python -m nltk.downloader stopwords
python -m nltk.downloader averaged_perceptron_tagger
python -m nltk.downloader wordnet

# Download spaCy model
python -m spacy download en_core_web_sm