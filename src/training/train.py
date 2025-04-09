import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import logging
from typing import Dict, Any, List, Tuple

from src.models.bias_correction_model import DeepBiasCorrectionModel
from src.data.weather_data_module import WeatherDataModule
from src.utils.normalization import save_normalization_params
from src.training.config import get_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a bias correction model.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing processed data")
    parser.add_argument("--accelerator", type=str, default="gpu", choices=["cpu", "gpu", "tpu"], help="Accelerator type")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--experiment_name", type=str, default="bias_correction", help="Experiment name for logging")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for model layers")
    parser.add_argument("--bidirectional", type=bool, default=True, help="Use bidirectional LSTM")
    return parser.parse_args()

def train_model(args):
    # Set random seed for reproducibility
    pl.seed_everything(42)
    logger.info("Seed set to 42")

    # Load configuration
    config = get_config()
    
    # Override config with command-line args
    config['data']['batch_size'] = args.batch_size
    config['model']['hidden_dim'] = args.hidden_dim
    config['model']['bidirectional'] = args.bidirectional
    config['training']['max_epochs'] = args.max_epochs
    
    # Initialize data module
    data_module = WeatherDataModule(
        data_dir=args.data_dir,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        sequence_length=config['data']['sequence_length']
    )
    
    # Initialize model
    model = DeepBiasCorrectionModel(
        input_dim=config['data']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        num_layers=config['model']['num_layers'],
        dropout_rate=config['model']['dropout_rate'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        physics_weight=config['training']['physics_weight'],
        bidirectional=config['model']['bidirectional']
    )
    
    # Print model architecture
    logger.info("Model architecture:")
    logger.info(f"- Input dimension: {config['data']['input_dim']}")
    logger.info(f"- Hidden dimension: {config['model']['hidden_dim']}")
    logger.info(f"- Output dimension: {config['model']['output_dim']}")
    logger.info(f"- Bidirectional LSTM: {config['model']['bidirectional']}")
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath=os.path.join(config['logging']['log_dir'], args.experiment_name, "checkpoints"),
        filename="bias_correction-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    
    early_stopping_callback = EarlyStopping(
        monitor="val/loss",
        patience=config['training']['patience'],
        mode="min",
    )
    
    # Set up logger
    tb_logger = TensorBoardLogger(
        save_dir=config['logging']['log_dir'],
        name=args.experiment_name,
        version="version_0"
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        max_epochs=config['training']['max_epochs'],
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=tb_logger,
        log_every_n_steps=config['logging']['log_every_n_steps'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        enable_progress_bar=True,
        precision=16 if args.accelerator == "gpu" else 32  # Use mixed precision on GPU
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    # Save normalization parameters
    data_module.setup()  # Ensure datasets are set up
    # Get the first dataset to extract normalization parameters
    first_dataset = data_module.train_datasets[0].dataset
    
    # Save normalization parameters
    norm_params_path = os.path.join(config['logging']['log_dir'], args.experiment_name, "normalization_params.json")
    save_normalization_params(
        target_mean=first_dataset.target_mean,
        target_std=first_dataset.target_std,
        feature_means=first_dataset.feature_means,
        feature_stds=first_dataset.feature_stds,
        save_path=norm_params_path
    )
    logger.info(f"Saved normalization parameters to {norm_params_path}")
    
    return trainer, model, data_module

def main():
    args = parse_args()
    trainer, model, data_module = train_model(args)
    
    # Test the model (optional)
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()