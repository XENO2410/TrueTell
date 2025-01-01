# src/train.py

import logging
import os
from model_trainer import ModelTrainer
import pandas as pd
import numpy as np
from datetime import datetime

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

def train_and_save_model():
    """Main function to train and save the model"""
    try:
        # Create necessary directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("datasets", exist_ok=True)
        
        # Create initial dataset if needed
        create_initial_dataset()
        
        # Initialize trainer
        logger.info("Initializing model trainer...")
        trainer = ModelTrainer()
        
        # Load and preprocess data
        logger.info("Loading and preprocessing dataset...")
        df = trainer.load_and_preprocess_data('datasets/factdata.csv')
        
        if df is not None:
            # Display dataset info
            logger.info(f"Dataset loaded successfully with {len(df)} samples")
            logger.info(f"Class distribution:\n{df['Fact Check'].value_counts()}")
            
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
                
                # Log performance metrics
                logger.info(f"Training accuracy: {metrics['train_accuracy']:.2%}")
                logger.info(f"Test accuracy: {metrics['test_accuracy']:.2%}")
                logger.info(f"Cross-validation mean: {metrics['cv_mean']:.2%}")
                
                logger.info("Model training completed successfully!")
                
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Starting model training process...")
    train_and_save_model()