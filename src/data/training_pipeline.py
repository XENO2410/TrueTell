# src/data/training_pipeline.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime
import logging
from tqdm import tqdm
import wandb
import os
import json

class MisinformationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class TrainingPipeline:
    def __init__(self, model_name: str = "bert-base-uncased", wandb_enabled: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.wandb_enabled = wandb_enabled
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2
        ).to(self.device)
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('TrainingPipeline')

    def prepare_data(self, 
                    df: pd.DataFrame, 
                    test_size: float = 0.2,
                    val_size: float = 0.1) -> Dict[str, DataLoader]:
        """Prepare data for training"""
        # Split data
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            df['text'].values, 
            df['label'].values,
            test_size=test_size,
            stratify=df['label'].values
        )

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts,
            train_labels,
            test_size=val_size,
            stratify=train_labels
        )

        # Create datasets
        train_dataset = MisinformationDataset(
            train_texts, 
            train_labels,
            self.tokenizer
        )
        
        val_dataset = MisinformationDataset(
            val_texts,
            val_labels,
            self.tokenizer
        )
        
        test_dataset = MisinformationDataset(
            test_texts,
            test_labels,
            self.tokenizer
        )

        # Create dataloaders
        return {
            'train': DataLoader(train_dataset, batch_size=16, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=32),
            'test': DataLoader(test_dataset, batch_size=32)
        }

    def train(self, 
              dataloaders: Dict[str, DataLoader],
              config: Dict = None) -> Dict:
        """Train the model"""
        if config is None:
            config = {
                'epochs': 5,
                'learning_rate': 2e-5,
                'batch_size': 16
            }

        # Initialize wandb if enabled
        if self.wandb_enabled:
            wandb.init(
                project="indian-misinfo-detection",
                config=config
            )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        
        best_val_loss = float('inf')
        training_stats = []

        for epoch in range(config['epochs']):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch in tqdm(dataloaders['train'], desc=f'Epoch {epoch + 1}/{config["epochs"]}'):
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                train_loss += loss.item()

                predictions = torch.argmax(outputs.logits, dim=1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)

                loss.backward()
                optimizer.step()

            # Validation
            val_stats = self.evaluate(dataloaders['val'])

            # Log metrics
            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': train_loss / len(dataloaders['train']),
                'train_accuracy': train_correct / train_total,
                'val_loss': val_stats['loss'],
                'val_accuracy': val_stats['accuracy']
            }
            
            if self.wandb_enabled:
                wandb.log(epoch_stats)
            
            self.logger.info(f"Epoch {epoch + 1} stats: {epoch_stats}")

            # Save best model
            if val_stats['loss'] < best_val_loss:
                best_val_loss = val_stats['loss']
                self.save_model(f'best_model_epoch_{epoch+1}')

            training_stats.append(epoch_stats)
            scheduler.step()

        # Final evaluation on test set
        test_stats = self.evaluate(dataloaders['test'])
        self.logger.info(f"Final Test Results: {test_stats}")

        if self.wandb_enabled:
            wandb.finish()

        return {
            'training_stats': training_stats,
            'test_stats': test_stats
        }

    def evaluate(self, dataloader: DataLoader) -> Dict:
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs.loss.item()
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total,
            'predictions': all_predictions,
            'labels': all_labels
        }

    def save_model(self, path: str):
        """Save the model"""
        os.makedirs('models', exist_ok=True)
        full_path = f"models/{path}"
        
        # Save model
        self.model.save_pretrained(full_path)
        self.tokenizer.save_pretrained(full_path)
        
        # Save configuration
        config = {
            'model_name': self.model_name,
            'save_date': datetime.now().isoformat()
        }
        
        with open(f"{full_path}/config.json", 'w') as f:
            json.dump(config, f)

    def load_model(self, path: str):
        """Load a saved model"""
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)