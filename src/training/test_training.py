import os
import logging
import argparse
from typing import Dict, Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from ..models import DeepBiasCorrectionModel
from ..data.weather_data_module import WeatherDataModule
from .train import test_model, parse_args

def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_test_data_module(
    config: Dict[str, Dict[str, Any]],
    data_dir: str
) -> pl.LightningDataModule:
    """Create a data module for testing."""
    data_config = config['data']
    
    return WeatherDataModule(
        data_dir=data_dir,
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers']
    )

def test_trained_model(checkpoint_path: str):
    """
    Test a trained model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
    """
    # Parse arguments
    args = parse_args()
    
    # Call the test_model function from train.py
    test_results = test_model(args, model_path=checkpoint_path)
    
    # Log results
    logger = setup_logging()
    logger.info("Test results:")
    for metric_name, metric_value in test_results[0].items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    return test_results

def main():
    """Test the model training pipeline."""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting test training pipeline...")
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test bias correction model')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to model checkpoint for testing')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                      help='Directory containing processed data')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    if args.checkpoint:
        # Test an existing trained model
        logger.info(f"Testing model from checkpoint: {args.checkpoint}")
        test_trained_model(args.checkpoint)
    else:
        # Run small-scale training and testing
        run_test_training(args.data_dir)
    
def run_test_training(data_dir):
    """Run a small-scale test training session."""
    logger = setup_logging()
    
    # Create model using the same architecture as in main training
    model = DeepBiasCorrectionModel(
        input_dim=7,  # Number of input features - temperature, humidity, wind speed, wind direction, cloud cover (3 layers)
        hidden_dim=64,
        output_dim=1,  # Temperature bias correction
        num_layers=2,
        dropout_rate=0.1,
        learning_rate=1e-4,
        weight_decay=1e-5,
        physics_weight=0.1,
        num_mc_samples=20,  # For uncertainty estimation
        bidirectional=True
    )
    logger.info("Created test model")
    
    # Print model architecture for debugging
    logger.info(f"Model architecture:")
    logger.info(f"- Input dimension: {model.input_dim}")
    logger.info(f"- Hidden dimension: {model.hidden_dim}")
    logger.info(f"- Output dimension: {model.output_dim}")
    logger.info(f"- LSTM bidirectional: {model.bidirectional if hasattr(model, 'bidirectional') else True}")
    
    # Create data module
    data_module = WeatherDataModule(
        data_dir=data_dir,
        batch_size=8,
        num_workers=2
    )
    logger.info("Created test data module")
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('logs', 'test', 'checkpoints'),
        filename='test-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val/loss',
        mode='min'
    )
    
    # Create logger
    tb_logger = TensorBoardLogger(
        save_dir='logs',
        name='test'
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=2,  # Just a few epochs for testing
        callbacks=[checkpoint_callback],
        logger=tb_logger,
        accelerator='auto',
        devices=1,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        limit_train_batches=0.1,  # Use only 10% of training data
        limit_val_batches=0.1,  # Use only 10% of validation data
        limit_test_batches=0.1  # Use only 10% of test data
    )
    
    # Train model
    logger.info("Starting test training...")
    trainer.fit(model, data_module)
    logger.info("Test training completed!")
    
    # Test model
    logger.info("Starting test evaluation...")
    trainer.test(model, data_module)
    logger.info("Test evaluation completed!")
    
    # Log final metrics
    metrics = trainer.callback_metrics
    logger.info("Final test metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")

if __name__ == '__main__':
    main() 