import os
import argparse
import logging
from typing import Dict, Any, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from src.models import DeepBiasCorrectionModel
from src.data.weather_data_module import WeatherDataModule

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train bias correction model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing processed data')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    
    # Model arguments
    parser.add_argument('--input_dim', type=int, default=7,  # Updated to match feature count
                      help='Dimension of input features')
    parser.add_argument('--hidden_dim', type=int, default=64,
                      help='Dimension of hidden layers')
    parser.add_argument('--num_layers', type=int, default=2,
                      help='Number of LSTM layers')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                      help='Dropout rate')
    parser.add_argument('--bidirectional', action='store_true',
                      help='Use bidirectional LSTM')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='Weight decay')
    parser.add_argument('--physics_weight', type=float, default=0.1,
                      help='Weight for physics-guided loss')
    parser.add_argument('--max_epochs', type=int, default=100,
                      help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10,
                      help='Patience for early stopping')
    parser.add_argument('--accelerator', type=str, default='auto',
                      choices=['cpu', 'gpu', 'auto'],
                      help='Accelerator type (cpu, gpu, or auto)')
    
    # Logging arguments
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory for logging')
    parser.add_argument('--experiment_name', type=str, default='default',
                      help='Name of the experiment')
    
    return parser.parse_args()

def setup_logging(args: argparse.Namespace) -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)

def create_model(args: argparse.Namespace) -> pl.LightningModule:
    """Create and return the model."""
    return DeepBiasCorrectionModel(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=1,  # For temperature prediction
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        physics_weight=args.physics_weight,
        bidirectional=args.bidirectional
    )

def create_callbacks(args: argparse.Namespace) -> list:
    """Create and return training callbacks."""
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, args.experiment_name, 'checkpoints'),
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val/loss',
        mode='min'
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val/loss',
        patience=args.patience,
        mode='min'
    )
    callbacks.append(early_stopping)
    
    return callbacks

def create_logger(args: argparse.Namespace) -> pl.loggers.Logger:
    """Create and return the logger."""
    return TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name
    )

def train_model(args):
    """Train the bias correction model.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Tuple of (trainer, model, data_module)
    """
    # Create data module
    data_module = WeatherDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model with correct architecture according to research paper
    model = DeepBiasCorrectionModel(
        input_dim=7,  # Number of input features - temperature, humidity, wind speed, wind direction, cloud cover (3 layers)
        hidden_dim=args.hidden_dim,
        output_dim=1,  # Temperature bias correction
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate if hasattr(args, 'learning_rate') else 1e-4,
        weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 1e-5,
        physics_weight=args.physics_weight if hasattr(args, 'physics_weight') else 0.1,
        num_mc_samples=20,  # Number of Monte Carlo dropout samples for uncertainty
        bidirectional=args.bidirectional if hasattr(args, 'bidirectional') else True
    )
    
    # Print model architecture summary for debugging
    print(f"Model architecture:")
    print(f"- Input dimension: {model.input_dim}")
    print(f"- Hidden dimension: {model.hidden_dim}")
    print(f"- Output dimension: {model.output_dim}")
    print(f"- Bidirectional LSTM: {getattr(args, 'bidirectional', True)}")
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, args.experiment_name, 'checkpoints'),
        filename='bias_correction-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val/loss',
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor='val/loss',
        patience=args.patience if hasattr(args, 'patience') else 10,
        mode='min'
    )
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name
    )
    
    # Create trainer with reduced verbosity
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=1,
        callbacks=[
            checkpoint_callback, 
            early_stopping,
            TQDMProgressBar(refresh_rate=1)  # Refresh rate of 1 makes it easier to read
        ],
        logger=logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        # Reduce verbosity
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=50  # Log less frequently
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Return trainer and data_module so they can be used for testing
    return trainer, model, data_module

def test_model(args, model_path=None):
    """
    Test a trained model.
    
    Args:
        args: Command line arguments
        model_path: Path to the trained model checkpoint
        
    Returns:
        test_results: Dictionary containing test metrics
    """
    # Create data module
    data_module = WeatherDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Load model from checkpoint if provided
    if model_path:
        model = DeepBiasCorrectionModel.load_from_checkpoint(model_path)
        print(f"Loaded model from checkpoint: {model_path}")
    else:
        # Create a new model with the same configuration as in train_model
        model = DeepBiasCorrectionModel(
            input_dim=7,  # Number of input features
            hidden_dim=args.hidden_dim,
            output_dim=1,  # Temperature bias correction
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate,
            learning_rate=args.learning_rate if hasattr(args, 'learning_rate') else 1e-4,
            weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 1e-5,
            physics_weight=args.physics_weight if hasattr(args, 'physics_weight') else 0.1,
            num_mc_samples=20,  # For uncertainty
            bidirectional=args.bidirectional if hasattr(args, 'bidirectional') else True
        )
        print("Created new model for testing")
    
    # Create trainer for testing
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=1,
        logger=False  # No logging needed for testing
    )
    
    # Run test
    print("Starting model testing...")
    test_results = trainer.test(model, datamodule=data_module)
    print("Testing completed")
    
    return test_results

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(args)
    logger = logging.getLogger(__name__)
    
    # Set random seed for reproducibility
    pl.seed_everything(42)
    logger.info("Seed set to 42")
    
    # Train model
    trainer, model, data_module = train_model(args)
    
    # Test model
    logger.info("Starting testing...")
    trainer.test(model, data_module)
    logger.info("Testing completed!")

if __name__ == '__main__':
    main() 