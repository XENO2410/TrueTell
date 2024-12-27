# src/data/validation.py

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from pathlib import Path

class ModelValidator:
    def __init__(self, save_dir: str = "validation_results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('ModelValidator')

    def validate_model(self, 
                      predictions: List[int], 
                      true_labels: List[int],
                      texts: List[str] = None) -> Dict:
        """Comprehensive model validation"""
        
        # Basic metrics
        report = classification_report(true_labels, predictions, output_dict=True)
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        # Error analysis
        errors = self._analyze_errors(predictions, true_labels, texts)
        
        # Create visualizations
        self._plot_confusion_matrix(conf_matrix)
        self._plot_metrics(report)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results = {
            'timestamp': timestamp,
            'metrics': report,
            'confusion_matrix': conf_matrix.tolist(),
            'error_analysis': errors
        }
        
        with open(self.save_dir / f'validation_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        return results

    def _analyze_errors(self, 
                       predictions: List[int], 
                       true_labels: List[int],
                       texts: List[str] = None) -> Dict:
        """Analyze prediction errors"""
        errors = {
            'false_positives': [],
            'false_negatives': [],
            'error_patterns': {}
        }
        
        if texts:
            for i, (pred, true) in enumerate(zip(predictions, true_labels)):
                if pred != true:
                    error_type = 'false_positives' if pred == 1 else 'false_negatives'
                    errors[error_type].append({
                        'text': texts[i],
                        'predicted': pred,
                        'true': true
                    })
        
        # Analyze error patterns
        errors['error_patterns'] = self._identify_error_patterns(errors)
        
        return errors

    def _identify_error_patterns(self, errors: Dict) -> Dict:
        """Identify common patterns in errors"""
        patterns = {
            'common_words': self._analyze_common_words(errors),
            'text_length': self._analyze_text_length(errors),
            'language_patterns': self._analyze_language_patterns(errors)
        }
        return patterns

    def _analyze_common_words(self, errors: Dict) -> Dict:
        """Analyze common words in errors"""
        # Add word analysis logic
        pass

    def _analyze_text_length(self, errors: Dict) -> Dict:
        """Analyze text length patterns in errors"""
        # Add length analysis logic
        pass

    def _analyze_language_patterns(self, errors: Dict) -> Dict:
        """Analyze language patterns in errors"""
        # Add language pattern analysis logic
        pass

    def _plot_confusion_matrix(self, conf_matrix: np.ndarray):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d',
            cmap='Blues'
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.save_dir / 'confusion_matrix.png')
        plt.close()

    def _plot_metrics(self, report: Dict):
        """Plot classification metrics"""
        metrics = ['precision', 'recall', 'f1-score']
        values = [report['weighted avg'][metric] for metric in metrics]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=metrics, y=values)
        plt.title('Classification Metrics')
        plt.ylim(0, 1)
        plt.savefig(self.save_dir / 'metrics.png')
        plt.close()

    def generate_report(self, results: Dict) -> str:
        """Generate validation report"""
        report = f"""
        Validation Report
        ================
        Timestamp: {results['timestamp']}
        
        Overall Metrics:
        --------------
        Accuracy: {results['metrics']['accuracy']:.3f}
        Weighted F1-Score: {results['metrics']['weighted avg']['f1-score']:.3f}
        
        Class-wise Performance:
        --------------------
        """
        
        for class_label in ['0', '1']:
            report += f"""
            Class {class_label}:
            - Precision: {results['metrics'][class_label]['precision']:.3f}
            - Recall: {results['metrics'][class_label]['recall']:.3f}
            - F1-Score: {results['metrics'][class_label]['f1-score']:.3f}
            """
        
        report += "\nError Analysis:\n"
        report += f"- False Positives: {len(results['error_analysis']['false_positives'])}\n"
        report += f"- False Negatives: {len(results['error_analysis']['false_negatives'])}\n"
        
        return report