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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc

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
            y_train_pred = self.pipeline.predict(X_train)
            y_test_pred = self.pipeline.predict(X_test)
            
            metrics = {
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'test_accuracy': accuracy_score(y_test, y_test_pred),
                'cv_mean': grid_search.best_score_,
                'cv_std': grid_search.cv_results_['std_test_score'][grid_search.best_index_],
                'classification_report': classification_report(y_test, y_test_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist()
            }
    
            # Update model info
            self.model_info.update({
                "version": self.model_version,
                "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_size": len(df),
                "best_params": grid_search.best_params_,
                "best_score": grid_search.best_score_,  # Add best score
                "feature_count": len(self.pipeline.named_steps['vectorizer'].get_feature_names_out())
            })
            
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            return metrics
                
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
            
    def _generate_visualizations(self, X_test, y_test, y_pred, y_pred_proba=None):
        """Generate and save model performance visualizations"""
        try:
            # Create visualizations directory
            viz_dir = "models/visualizations"
            os.makedirs(viz_dir, exist_ok=True)
            
            # 1. Confusion Matrix Heatmap
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Fake', 'True', 'Unverified'],
                        yticklabels=['Fake', 'True', 'Unverified'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(f'{viz_dir}/confusion_matrix.png')
            plt.close()
    
            if y_pred_proba is not None:
                # 2. ROC Curves
                plt.figure(figsize=(10, 8))
                n_classes = len(np.unique(y_test))
                
                if n_classes == 2:
                    # Binary classification
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    
                    plt.plot(fpr, tpr, color='darkorange', lw=2,
                            label=f'ROC curve (AUC = {roc_auc:.2f})')
                else:
                    # Multi-class classification
                    y_test_bin = label_binarize(y_test, classes=range(n_classes))
                    for i in range(n_classes):
                        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                        roc_auc = auc(fpr, tpr)
                        plt.plot(fpr, tpr, lw=2,
                                label=f'ROC class {i} (AUC = {roc_auc:.2f})')
                
                plt.plot([0, 1], [0, 1], 'k--', lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curves')
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(f'{viz_dir}/roc_curves.png')
                plt.close()
    
                # 3. Prediction Confidence Distribution
                plt.figure(figsize=(10, 6))
                confidence_scores = np.max(y_pred_proba, axis=1)
                sns.histplot(confidence_scores, bins=20)
                plt.title('Prediction Confidence Distribution')
                plt.xlabel('Confidence Score')
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig(f'{viz_dir}/confidence_distribution.png')
                plt.close()
    
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
    
    def _calculate_detailed_metrics(self, X_train, X_test, y_train, y_test):
        """Calculate comprehensive model metrics"""
        try:
            # Get predictions
            y_train_pred = self.pipeline.predict(X_train)
            y_test_pred = self.pipeline.predict(X_test)
            y_test_proba = self.pipeline.predict_proba(X_test)
            
            # Generate visualizations
            self._generate_visualizations(X_test, y_test, y_test_pred, y_test_proba)
            
            # Calculate metrics
            metrics = {
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'test_accuracy': accuracy_score(y_test, y_test_pred),
                'cv_scores': cross_val_score(self.pipeline, X_train, y_train, cv=5),
                'classification_report': classification_report(y_test, y_test_pred, 
                                                            output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist(),
                'feature_importance': self._get_feature_importance()
            }
            
            # Add class-wise metrics
            metrics['class_metrics'] = {}
            for class_label in np.unique(y_test):
                mask = y_test == class_label
                metrics['class_metrics'][f'class_{class_label}'] = {
                    'accuracy': accuracy_score(y_test[mask], y_test_pred[mask]),
                    'samples': sum(mask)
                }
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return None
    
    def _get_feature_importance(self):
        """Get feature importance scores"""
        try:
            vectorizer = self.pipeline.named_steps['vectorizer']
            classifier = self.pipeline.named_steps['classifier']
            
            feature_names = vectorizer.get_feature_names_out()
            importance_scores = classifier.feature_importances_
            
            # Get top 20 features
            top_indices = importance_scores.argsort()[-20:][::-1]
            
            return {
                'features': feature_names[top_indices].tolist(),
                'scores': importance_scores[top_indices].tolist()
            }
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return None