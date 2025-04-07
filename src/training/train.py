import os
import argparse
import logging
from typing import Dict, Any, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from ..models import DeepBiasCorrectionModel
from ..data import DataAligner

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
    parser.add_argument('--input_dim', type=int, default=5,
                      help='Dimension of input features')
    parser.add_argument('--hidden_dim', type=int, default=256,
                      help='Dimension of hidden layers')
    parser.add_argument('--num_layers', type=int, default=3,
                      help='Number of LSTM layers')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                      help='Dropout rate')
    parser.add_argument('--bidirectional', action='store_true',
                      help='Use bidirectional LSTM')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='Weight decay')
    parser.add_argument('--physics_weight', type=float, default=0.1,
                      help='Weight for physics-guided loss')
    parser.add_argument('--max_epochs', type=int, default=100,
                      help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10,
                      help='Patience for early stopping')
    
    # Logging arguments
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory for logging')
    parser.add_argument('--experiment_name', type=str, required=True,
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

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(args)
    logger = logging.getLogger(__name__)
    
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    # Create model
    model = create_model(args)
    logger.info("Created model with architecture:")
    logger.info(model)
    
    # Create data module
    data_module = DataAligner(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create callbacks
    callbacks = create_callbacks(args)
    
    # Create logger
    tb_logger = create_logger(args)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=tb_logger,
        accelerator='auto',
        devices=1,
        precision=32,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.fit(model, data_module)
    logger.info("Training completed!")
    
    # Test model
    logger.info("Starting testing...")
    trainer.test(model, data_module)
    logger.info("Testing completed!")

if __name__ == '__main__':
    main() 