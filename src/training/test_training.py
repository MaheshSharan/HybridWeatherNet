import os
import logging
import argparse
from typing import Dict, Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from ..models import DeepBiasCorrectionModel
from ..data import DataAligner
from .config import get_config

def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_test_model(config: Dict[str, Dict[str, Any]]) -> pl.LightningModule:
    """Create a model for testing."""
    model_config = config['model']
    training_config = config['training']
    
    return DeepBiasCorrectionModel(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        output_dim=model_config['output_dim'],
        num_layers=model_config['num_layers'],
        dropout_rate=model_config['dropout_rate'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        physics_weight=training_config['physics_weight'],
        bidirectional=model_config['bidirectional']
    )

def create_test_data_module(
    config: Dict[str, Dict[str, Any]],
    data_dir: str
) -> pl.LightningDataModule:
    """Create a data module for testing."""
    data_config = config['data']
    
    return DataAligner(
        data_dir=data_dir,
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers']
    )

def main():
    """Test the model training pipeline."""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting test training pipeline...")
    
    # Get configuration
    config = get_config()
    
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    # Create model
    model = create_test_model(config)
    logger.info("Created test model with architecture:")
    logger.info(model)
    
    # Create data module
    data_dir = os.path.join('data', 'processed')
    data_module = create_test_data_module(config, data_dir)
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
        precision=32,
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