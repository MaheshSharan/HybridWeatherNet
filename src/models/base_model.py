# src/models/base_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

class BiasCorrectionModel(pl.LightningModule):
    """
    Base class for bias correction models.
    
    This class defines the interface and common functionality for all bias correction models.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 1,
        learning_rate: float = 1e-4,
        dropout_rate: float = 0.2,
        weight_decay: float = 1e-5,
        physics_weight: float = 0.1
    ):
        """
        Initialize the bias correction model.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layers
            output_dim (int): Dimension of output (bias prediction)
            learning_rate (float): Learning rate for optimizer
            dropout_rate (float): Dropout rate for regularization
            weight_decay (float): Weight decay for L2 regularization
            physics_weight (float): Weight for physics-guided loss term
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.physics_weight = physics_weight
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Initialize model components (to be implemented by subclasses)
        self.temporal_module = None
        self.spatial_module = None
        self.fusion_layer = None
        self.output_layer = None
        
        # Initialize uncertainty estimation (to be implemented by subclasses)
        self.uncertainty_estimator = None
    
    def forward(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            return_uncertainty (bool): Whether to return uncertainty estimates
            
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: 
                - If return_uncertainty is False: Predicted bias
                - If return_uncertainty is True: Tuple of (predicted bias, uncertainty)
        """
        raise NotImplementedError("Forward method must be implemented by subclasses")
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch (Dict[str, torch.Tensor]): Batch of data
            batch_idx (int): Batch index
            
        Returns:
            torch.Tensor: Loss value
        """
        # Extract data from batch
        x = batch['input']
        y = batch['target']
        
        # Forward pass
        y_pred = self(x)
        
        # Calculate MSE loss
        mse_loss = F.mse_loss(y_pred, y)
        
        # Calculate physics-guided loss (to be implemented by subclasses)
        physics_loss = self.physics_guided_loss(x, y_pred, y)
        
        # Combine losses
        total_loss = mse_loss + self.physics_weight * physics_loss
        
        # Log losses
        self.log('train/mse_loss', mse_loss, prog_bar=True)
        self.log('train/physics_loss', physics_loss, prog_bar=True)
        self.log('train/total_loss', total_loss, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.
        
        Args:
            batch (Dict[str, torch.Tensor]): Batch of data
            batch_idx (int): Batch index
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of validation metrics
        """
        # Extract data from batch
        x = batch['input']
        y = batch['target']
        
        # Forward pass
        y_pred = self(x)
        
        # Calculate MSE loss
        mse_loss = F.mse_loss(y_pred, y)
        
        # Calculate physics-guided loss
        physics_loss = self.physics_guided_loss(x, y_pred, y)
        
        # Combine losses
        total_loss = mse_loss + self.physics_weight * physics_loss
        
        # Calculate metrics
        mae = F.l1_loss(y_pred, y)
        rmse = torch.sqrt(mse_loss)
        
        # Log metrics
        self.log('val/mse_loss', mse_loss, prog_bar=True)
        self.log('val/physics_loss', physics_loss, prog_bar=True)
        self.log('val/total_loss', total_loss, prog_bar=True)
        self.log('val/mae', mae, prog_bar=True)
        self.log('val/rmse', rmse, prog_bar=True)
        
        return {
            'val/mse_loss': mse_loss,
            'val/physics_loss': physics_loss,
            'val/total_loss': total_loss,
            'val/mae': mae,
            'val/rmse': rmse
        }
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step.
        
        Args:
            batch (Dict[str, torch.Tensor]): Batch of data
            batch_idx (int): Batch index
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of test metrics
        """
        # Extract data from batch
        x = batch['input']
        y = batch['target']
        
        # Forward pass with uncertainty
        y_pred, uncertainty = self(x, return_uncertainty=True)
        
        # Calculate MSE loss
        mse_loss = F.mse_loss(y_pred, y)
        
        # Calculate physics-guided loss
        physics_loss = self.physics_guided_loss(x, y_pred, y)
        
        # Combine losses
        total_loss = mse_loss + self.physics_weight * physics_loss
        
        # Calculate metrics
        mae = F.l1_loss(y_pred, y)
        rmse = torch.sqrt(mse_loss)
        
        # Calculate calibration metrics (to be implemented by subclasses)
        calibration_metrics = self.calculate_calibration_metrics(y, y_pred, uncertainty)
        
        # Log metrics
        self.log('test/mse_loss', mse_loss, prog_bar=True)
        self.log('test/physics_loss', physics_loss, prog_bar=True)
        self.log('test/total_loss', total_loss, prog_bar=True)
        self.log('test/mae', mae, prog_bar=True)
        self.log('test/rmse', rmse, prog_bar=True)
        
        for metric_name, metric_value in calibration_metrics.items():
            self.log(f'test/{metric_name}', metric_value, prog_bar=True)
        
        return {
            'test/mse_loss': mse_loss,
            'test/physics_loss': physics_loss,
            'test/total_loss': total_loss,
            'test/mae': mae,
            'test/rmse': rmse,
            **calibration_metrics
        }
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizers.
        
        Returns:
            torch.optim.Optimizer: Optimizer
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/mse_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def physics_guided_loss(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate physics-guided loss.
        
        This method should be implemented by subclasses to incorporate
        domain-specific physical constraints into the loss function.
        
        Args:
            x (torch.Tensor): Input tensor
            y_pred (torch.Tensor): Predicted bias
            y_true (torch.Tensor): True bias
            
        Returns:
            torch.Tensor: Physics-guided loss
        """
        # Default implementation: spatial smoothness
        # Encourage neighboring stations to have similar bias corrections
        if hasattr(self, 'station_coords') and self.station_coords is not None:
            # Calculate pairwise distances between stations
            coords = self.station_coords
            dist = torch.cdist(coords, coords)
            
            # Calculate pairwise differences in predictions
            diff = torch.cdist(y_pred, y_pred)
            
            # Weight differences by inverse distance
            weights = 1.0 / (dist + 1e-6)  # Add small constant to avoid division by zero
            weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize
            
            # Calculate weighted sum of squared differences
            smoothness_loss = (weights * diff.pow(2)).sum()
            
            return smoothness_loss
        
        # If no station coordinates are available, return zero loss
        return torch.tensor(0.0, device=y_pred.device)
    
    def calculate_calibration_metrics(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        uncertainty: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate calibration metrics for uncertainty estimates.
        
        Args:
            y_true (torch.Tensor): True values
            y_pred (torch.Tensor): Predicted values
            uncertainty (torch.Tensor): Uncertainty estimates
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of calibration metrics
        """
        # Debug prints
        print(f"y_true shape: {y_true.shape}")
        print(f"y_pred shape: {y_pred.shape}")
        print(f"uncertainty shape: {uncertainty.shape}")
        
        # Check for NaN values
        if torch.isnan(y_true).any():
            print("Warning: NaN values in y_true")
            y_true = torch.nan_to_num(y_true, nan=0.0)
        if torch.isnan(y_pred).any():
            print("Warning: NaN values in y_pred")
            y_pred = torch.nan_to_num(y_pred, nan=0.0)
        if torch.isnan(uncertainty).any():
            print("Warning: NaN values in uncertainty")
            uncertainty = torch.nan_to_num(uncertainty, nan=1.0)  # Use 1.0 for NaN uncertainty
        
        # Ensure uncertainty is positive and non-zero
        uncertainty = torch.clamp(uncertainty, min=1e-6)
        
        # Reshape tensors to 2D for correlation calculation
        y_true_flat = y_true.reshape(-1, y_true.size(-1))
        y_pred_flat = y_pred.reshape(-1, y_pred.size(-1))
        uncertainty_flat = uncertainty.reshape(-1, uncertainty.size(-1))
        
        # Calculate normalized error
        normalized_error = (y_true_flat - y_pred_flat) / uncertainty_flat
        
        # Calculate absolute errors
        abs_errors = torch.abs(y_true_flat - y_pred_flat)
        
        # Calculate calibration metrics with error handling
        try:
            # Calculate correlation coefficient
            stacked = torch.stack([
                abs_errors.mean(dim=-1),  # Average across last dimension
                uncertainty_flat.mean(dim=-1)  # Average across last dimension
            ])
            
            # Check if we have enough samples for correlation
            if stacked.shape[1] < 2:
                print("Warning: Not enough samples for correlation calculation")
                corr = torch.tensor(0.0, device=stacked.device)
            else:
                corr = torch.corrcoef(stacked)[0, 1]
                if torch.isnan(corr):
                    print("Warning: Correlation coefficient is NaN")
                    corr = torch.tensor(0.0, device=stacked.device)
            
            metrics = {
                'calibration_error': torch.mean(torch.abs(normalized_error - 0)),
                'uncertainty_correlation': corr
            }
        except Exception as e:
            print(f"Error in calibration metrics calculation: {str(e)}")
            metrics = {
                'calibration_error': torch.tensor(0.0, device=y_true.device),
                'uncertainty_correlation': torch.tensor(0.0, device=y_true.device)
            }
        
        return metrics 