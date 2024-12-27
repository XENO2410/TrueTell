# src/data/data_augmentation.py

import random
import nltk
from typing import List, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

class DataAugmenter:
    def __init__(self):
        self.setup_logging()
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('DataAugmenter')

    def augment_dataset(self, 
                       df: pd.DataFrame, 
                       techniques: List[str] = None,
                       multiplier: int = 2) -> pd.DataFrame:
        """
        Augment the dataset using basic text manipulation techniques
        """
        if techniques is None:
            techniques = ['swap_words', 'delete_words']

        augmented_data = []
        
        # Keep original data
        augmented_data.extend(df.to_dict('records'))
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            for _ in range(multiplier - 1):
                aug_row = row.copy()
                text = row['text']
                
                for technique in techniques:
                    if technique == 'swap_words':
                        aug_text = self._swap_words(text)
                    elif technique == 'delete_words':
                        aug_text = self._delete_words(text)
                    else:
                        aug_text = text
                        
                    aug_row['text'] = aug_text
                    aug_row['augmentation'] = technique
                    augmented_data.append(aug_row)

        return pd.DataFrame(augmented_data)

    def _swap_words(self, text: str) -> str:
        """Randomly swap some words in the text"""
        words = text.split()
        if len(words) <= 1:
            return text
            
        num_swaps = max(1, len(words) // 10)  # Swap about 10% of words
        for _ in range(num_swaps):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
            
        return ' '.join(words)

    def _delete_words(self, text: str) -> str:
        """Randomly delete some words from the text"""
        words = text.split()
        if len(words) <= 1:
            return text
            
        num_to_delete = max(1, len(words) // 10)  # Delete about 10% of words
        indices_to_delete = random.sample(range(len(words)), num_to_delete)
        
        return ' '.join([word for i, word in enumerate(words) if i not in indices_to_delete])

    def augment_regional(self, text: str, language: str) -> List[str]:
        """Augment text for regional variations"""
        augmented = []
        
        try:
            # Transliteration for Hindi
            if language == 'hi':
                dev_text = transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
                augmented.append(dev_text)
            
            # Add basic word variations
            words = text.split()
            if len(words) > 1:
                # Reverse word order
                augmented.append(' '.join(words[::-1]))
                
                # Remove stop words
                stop_words = set(['the', 'a', 'an', 'and', 'or', 'but'])
                filtered_text = ' '.join([w for w in words if w.lower() not in stop_words])
                augmented.append(filtered_text)
            
        except Exception as e:
            self.logger.error(f"Error in regional augmentation: {e}")
        
        return augmented

    def create_adversarial_examples(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create simple adversarial examples"""
        adversarial_data = []
        
        for _, row in df.iterrows():
            try:
                # Create variations of the text
                text = row['text']
                adv_texts = [
                    self._swap_words(text),
                    self._delete_words(text),
                    text.upper(),  # All caps version
                    text.lower(),  # All lowercase version
                ]
                
                for adv_text in adv_texts:
                    adversarial_data.append({
                        'text': adv_text,
                        'original_text': text,
                        'label': row['label'],
                        'type': 'adversarial'
                    })
                    
            except Exception as e:
                self.logger.error(f"Error creating adversarial example: {e}")
                continue
                
        return pd.DataFrame(adversarial_data)