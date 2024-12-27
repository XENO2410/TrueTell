# src/train_model.py

import asyncio
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from data.dataset_builder import IndianDatasetBuilder
from data.data_augmentation import DataAugmenter
from data.training_pipeline import TrainingPipeline
from data.validation import ModelValidator

# Load environment variables
load_dotenv()

def create_sample_dataset():
    """Create a small sample dataset for testing"""
    data = {
        'text': [
            "This is a true news article about India's economy",
            "False information about COVID-19 in Delhi",
            "Verified report about election results",
            "Misleading claims about vaccination drive",
            "Factual coverage of cricket match",
            # Add more examples
            "Real news about technology development in Bangalore",
            "Fake story about UFO sightings in Mumbai",
            "Accurate report about environmental initiatives",
            "Misleading statistics about education system",
            "True story about agricultural innovations"
        ],
        'label': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]  # 1 for true, 0 for false
    }
    return pd.DataFrame(data)

def setup_wandb():
    """Setup Weights & Biases configuration"""
    try:
        import wandb
        wandb_api_key = os.getenv('WANDB_API_KEY')
        if not wandb_api_key:
            print("Warning: WANDB_API_KEY not found in .env file")
            return False
        
        wandb.login(key=wandb_api_key)
        return True
    except Exception as e:
        print(f"Error setting up wandb: {e}")
        return False

async def main():
    try:
        print("Starting data collection and model training pipeline...")

        # Setup wandb
        wandb_enabled = setup_wandb()
        if not wandb_enabled:
            print("Continuing without wandb logging...")

        # Create sample dataset instead of scraping
        print("Creating sample dataset...")
        dataset = create_sample_dataset()
        print(f"Initial dataset size: {len(dataset)}")

        # Initialize augmenter
        print("Initializing data augmenter...")
        augmenter = DataAugmenter()
        
        # Augment data
        print("Augmenting dataset...")
        augmented_dataset = augmenter.augment_dataset(dataset)
        print(f"Augmented dataset size: {len(augmented_dataset)}")

        # Train Model
        print("Initializing training pipeline...")
        pipeline = TrainingPipeline(
            model_name="bert-base-uncased",
            wandb_enabled=wandb_enabled
        )
        
        print("Preparing data loaders...")
        dataloaders = pipeline.prepare_data(augmented_dataset)
        
        print("Starting model training...")
        training_config = {
            'epochs': 5,
            'learning_rate': 2e-5,
            'batch_size': 16,
            'model_name': "bert-base-uncased"
        }
        
        results = pipeline.train(
            dataloaders=dataloaders,
            config=training_config
        )

        # Validate Model
        print("Validating model...")
        validator = ModelValidator()
        validation_results = validator.validate_model(
            predictions=results['test_stats']['predictions'],
            true_labels=results['test_stats']['labels']
        )

        # Generate and Save Report
        print("Generating validation report...")
        report = validator.generate_report(validation_results)
        
        # Save report to file
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        report_path = f'reports/validation_report_{timestamp}.txt'
        os.makedirs('reports', exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Pipeline completed successfully! Report saved to {report_path}")
        print("\nValidation Report Summary:")
        print(report)

    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        raise e

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())