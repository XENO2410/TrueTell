# src/train.py

import logging
import os
from model_trainer import ModelTrainer
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging():
    """Setup logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_initial_dataset():
    """Create a basic dataset if none exists"""
    if not os.path.exists("datasets/factdata.csv"):
        logger.info("Creating initial dataset...")
        initial_data = {
            'Claims': [
                "This is a verified factual claim.",
                "This is misleading information.",
                "This needs verification.",
                # Add more example data as needed
            ],
            'Fact Check': [
                'True',
                'Fake',
                'Unverified',
                # Corresponding labels
            ]
        }
        df = pd.DataFrame(initial_data)
        os.makedirs("datasets", exist_ok=True)
        df.to_csv("datasets/factdata.csv", index=False)
        logger.info("Initial dataset created successfully")

def plot_training_progress(scores, trees):
    """Plot training progress"""
    plt.figure(figsize=(10, 6))
    plt.plot(trees, scores, marker='o')
    plt.title('Model Performance vs Number of Trees')
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy Score')
    plt.grid(True)
    
    # Save plot
    os.makedirs("models/visualizations", exist_ok=True)
    plt.savefig("models/visualizations/training_progress.png")
    plt.close()

def plot_class_distribution(df):
    """Plot class distribution"""
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Fact Check')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    # Save plot
    os.makedirs("models/visualizations", exist_ok=True)
    plt.savefig("models/visualizations/class_distribution.png")
    plt.close()

def train_and_save_model():
    """Main function to train and save the model with enhanced monitoring"""
    try:
        # Create necessary directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("datasets", exist_ok=True)
        os.makedirs("models/visualizations", exist_ok=True)
        
        # Create initial dataset if needed
        create_initial_dataset()
        
        # Initialize trainer
        logger.info("Initializing model trainer...")
        trainer = ModelTrainer()
        
        # Load and preprocess data
        logger.info("Loading and preprocessing dataset...")
        df = trainer.load_and_preprocess_data('datasets/factdata.csv')
        
        if df is not None:
            # Display and plot dataset info
            logger.info(f"Dataset loaded successfully with {len(df)} samples")
            class_dist = df['Fact Check'].value_counts()
            logger.info(f"Class distribution:\n{class_dist}")
            
            # Plot class distribution
            plot_class_distribution(df)
            
            # Train model
            logger.info("Training model...")
            metrics = trainer.train_model(df, test_size=0.2)
            
            if metrics:
                # Save model
                logger.info("Saving model...")
                trainer.save_model("models/misinformation_detector.joblib")
                
                # Export metrics
                logger.info("Exporting metrics...")
                trainer.export_metrics("models/model_metrics.json")
                
                # Log detailed performance metrics
                logger.info("\nDetailed Performance Metrics:")
                logger.info("-" * 50)
                
                # Model Info
                if trainer.model_info:
                    if 'best_score' in trainer.model_info:
                        logger.info(f"Best Model Score: {trainer.model_info['best_score']:.4f}")
                    if 'feature_count' in trainer.model_info:
                        logger.info(f"Feature Count: {trainer.model_info['feature_count']}")
                    if 'best_params' in trainer.model_info:
                        logger.info("\nBest Parameters:")
                        for param, value in trainer.model_info['best_params'].items():
                            logger.info(f"{param}: {value}")
                
                # Basic metrics
                logger.info("\nBasic Metrics:")
                logger.info(f"Training accuracy: {metrics['train_accuracy']:.2%}")
                logger.info(f"Test accuracy: {metrics['test_accuracy']:.2%}")
                logger.info(f"Cross-validation mean: {metrics['cv_mean']:.2%}")
                if 'cv_std' in metrics:
                    logger.info(f"Cross-validation std: {metrics['cv_std']:.2%}")
                
                # Classification Report
                if 'classification_report' in metrics:
                    logger.info("\nClassification Report:")
                    for class_name, values in metrics['classification_report'].items():
                        if isinstance(values, dict):
                            logger.info(f"\nClass {class_name}:")
                            logger.info(f"Precision: {values['precision']:.4f}")
                            logger.info(f"Recall: {values['recall']:.4f}")
                            logger.info(f"F1-score: {values['f1-score']:.4f}")
                
                # Confusion Matrix
                if 'confusion_matrix' in metrics:
                    logger.info("\nConfusion Matrix:")
                    logger.info(np.array(metrics['confusion_matrix']))
                
                logger.info("\nModel training completed successfully!")
                
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise

def main():
    """Main execution function"""
    try:
        global logger
        logger = setup_logging()
        logger.info("Starting model training process...")
        
        # Train and save model
        train_and_save_model()
        
        logger.info("Training process completed successfully!")
        
    except Exception as e:
        logger.error(f"Training process failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()