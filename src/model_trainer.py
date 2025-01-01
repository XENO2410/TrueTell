# src/model_trainer.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import logging
import json
from datetime import datetime
import os

class ModelTrainer:
    def __init__(self):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Model version tracking
        self.model_version = "1.0.0"
        self.model_info = {
            "version": self.model_version,
            "training_date": None,
            "dataset_size": None,
            "best_params": None,
            "feature_count": None
        }
        
        # Initialize pipeline
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.95
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            ))
        ])
        
        # Parameter grid for optimization
        self.param_grid = {
            'vectorizer__max_features': [3000, 5000],
            'vectorizer__ngram_range': [(1, 1), (1, 2)],
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [20, None],
            'classifier__min_samples_split': [2, 5]
        }
        
        self.model_metrics = {}
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess_data(self, filepath):
        """Load and preprocess the dataset"""
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            
            # Basic preprocessing
            df['Claims'] = df['Claims'].fillna('')
            df['Claims'] = df['Claims'].str.strip().str.lower()
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['Claims'])
            
            # Encode labels
            df['Fact Check'] = self.label_encoder.fit_transform(df['Fact Check'])
            
            self.logger.info(f"Loaded dataset with {len(df)} samples")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return None
            
    def train_model(self, df, test_size=0.2):
        """Train the model"""
        try:
            X = df['Claims']
            y = df['Fact Check']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=42, 
                stratify=y
            )
            
            # Train model with grid search
            self.logger.info("Starting GridSearchCV...")
            grid_search = GridSearchCV(
                self.pipeline,
                self.param_grid,
                cv=5,
                scoring='f1_macro',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Save best model
            self.pipeline = grid_search.best_estimator_
            
            # Calculate metrics
            self.model_metrics = self._calculate_metrics(
                X_train, X_test, y_train, y_test
            )

            # Update model info
            self.model_info.update({
                "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_size": len(df),
                "best_params": grid_search.best_params_,
                "feature_count": len(self.pipeline.named_steps['vectorizer'].get_feature_names_out())
            })
            
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            return self.model_metrics
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            return None

    def _calculate_metrics(self, X_train, X_test, y_train, y_test):
        """Calculate model performance metrics"""
        metrics = {}
        
        # Calculate predictions
        y_train_pred = self.pipeline.predict(X_train)
        y_test_pred = self.pipeline.predict(X_test)
        
        # Basic metrics
        metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
        metrics['test_accuracy'] = accuracy_score(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=5)
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Detailed metrics
        metrics['classification_report'] = classification_report(
            y_test, y_test_pred, output_dict=True
        )
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_test_pred)
        
        return metrics

    def save_model(self, filepath):
        """Save the trained model"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            joblib.dump(self.pipeline, filepath)
            
            # Save model info
            info_filepath = os.path.join(os.path.dirname(filepath), 'model_info.json')
            with open(info_filepath, 'w') as f:
                json.dump(self.model_info, f, indent=4)
                
            self.logger.info(f"Model saved successfully to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        try:
            self.pipeline = joblib.load(filepath)
            
            # Load model info
            info_filepath = os.path.join(os.path.dirname(filepath), 'model_info.json')
            if os.path.exists(info_filepath):
                with open(info_filepath, 'r') as f:
                    self.model_info = json.load(f)
                    
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
    
    def predict(self, text):
        """Make predictions"""
        try:
            if not isinstance(text, str):
                raise ValueError("Input must be a string")
            
            if len(text.strip()) == 0:
                raise ValueError("Input text cannot be empty")
                
            # Preprocess
            text = text.lower().strip()
            
            # Predict
            prediction = self.pipeline.predict([text])[0]
            probabilities = self.pipeline.predict_proba([text])[0]
            
            return {
                'prediction': prediction,
                'probabilities': probabilities,
                'confidence': probabilities.max()
            }
            
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            return None
            
    def export_metrics(self, filepath):
        """Export model metrics"""
        try:
            metrics_export = {
                "model_info": self.model_info,
                "performance_metrics": {
                    "accuracy": {
                        "train": self.model_metrics['train_accuracy'],
                        "test": self.model_metrics['test_accuracy'],
                        "cv_mean": self.model_metrics['cv_mean']
                    },
                    "classification_report": self.model_metrics['classification_report']
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(metrics_export, f, indent=4)
                
            self.logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {str(e)}")