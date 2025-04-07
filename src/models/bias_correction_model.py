import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

from .base_model import BiasCorrectionModel
from .lstm_module import LSTMModule
from .graph_module import GraphModule
from .attention_module import AttentionModule

class DeepBiasCorrectionModel(BiasCorrectionModel):
    """
    Deep learning model for weather forecast bias correction.
    
    This model combines:
    1. LSTM for temporal pattern learning
    2. Graph Neural Network for spatial relationships
    3. Attention mechanism for feature fusion
    4. Monte Carlo dropout for uncertainty estimation
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 1,
        num_layers: int = 3,
        dropout_rate: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        physics_weight: float = 0.1,
        num_mc_samples: int = 20,
        bidirectional: bool = True
    ):
        """
        Initialize the bias correction model.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layers
            output_dim (int): Dimension of output (usually 1 for temperature)
            num_layers (int): Number of LSTM/GNN layers
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
            weight_decay (float): Weight decay for regularization
            physics_weight (float): Weight for physics-guided loss
            num_mc_samples (int): Number of Monte Carlo samples for uncertainty
            bidirectional (bool): Whether to use bidirectional LSTM
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
            physics_weight=physics_weight
        )
        
        # LSTM module for temporal pattern learning
        self.lstm = LSTMModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional
        )
        
        # Graph Neural Network module for spatial relationships
        self.gnn = GraphModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )
        
        # Attention module for feature fusion
        self.attention = AttentionModule(
            input_dim=hidden_dim * 2,  # Combined LSTM and GNN outputs
            hidden_dim=hidden_dim,
            num_heads=8,
            dropout_rate=dropout_rate
        )
        
        # Output projection layer
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Monte Carlo dropout for uncertainty estimation
        self.num_mc_samples = num_mc_samples
        self.mc_dropout = nn.Dropout(p=dropout_rate)
        
        # Save hyperparameters
        self.save_hyperparameters()
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            edge_index (Optional[torch.Tensor]): Graph connectivity
            edge_attr (Optional[torch.Tensor]): Edge features
            return_uncertainty (bool): Whether to return uncertainty estimates
            
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - If return_uncertainty is False: Predicted bias
                - If return_uncertainty is True: Tuple of (predicted bias, uncertainty)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # GNN forward pass (if edge information is provided)
        if edge_index is not None:
            gnn_out = self.gnn(x.view(-1, x.size(-1)), edge_index, edge_attr)
            gnn_out = gnn_out.view(x.size(0), x.size(1), -1)
        else:
            gnn_out = torch.zeros_like(lstm_out)
        
        # Concatenate LSTM and GNN outputs
        combined = torch.cat([lstm_out, gnn_out], dim=-1)
        
        # Apply attention
        attended, attention_weights = self.attention(combined)
        
        # Project to output dimension
        output = self.output_proj(attended)
        
        if return_uncertainty:
            # Monte Carlo dropout for uncertainty estimation
            mc_outputs = []
            for _ in range(self.num_mc_samples):
                mc_out = self.mc_dropout(attended)
                mc_out = self.output_proj(mc_out)
                mc_outputs.append(mc_out)
            
            # Stack MC samples
            mc_outputs = torch.stack(mc_outputs, dim=0)
            
            # Calculate mean and variance
            mean = mc_outputs.mean(dim=0)
            variance = mc_outputs.var(dim=0)
            
            return mean, variance
        
        return output
    
    def estimate_uncertainty(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate uncertainty using Monte Carlo dropout.
        
        Args:
            x (torch.Tensor): Input tensor
            edge_index (Optional[torch.Tensor]): Graph connectivity
            edge_attr (Optional[torch.Tensor]): Edge features
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and variance of predictions
        """
        self.train()  # Enable dropout
        with torch.no_grad():
            mean, variance = self(x, edge_index, edge_attr, return_uncertainty=True)
        self.eval()  # Disable dropout
        return mean, variance
    
    def calculate_physics_loss(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate physics-guided loss.
        
        Args:
            x (torch.Tensor): Input features
            y_pred (torch.Tensor): Predicted bias
            y_true (torch.Tensor): True bias
            
        Returns:
            torch.Tensor: Physics-guided loss
        """
        # Spatial smoothness loss
        spatial_diff = y_pred[:, 1:] - y_pred[:, :-1]
        spatial_smoothness = torch.mean(torch.abs(spatial_diff))
        
        # Temporal consistency loss
        temporal_diff = y_pred[:, :, 1:] - y_pred[:, :, :-1]
        temporal_consistency = torch.mean(torch.abs(temporal_diff))
        
        # Physical constraints (e.g., temperature range)
        physical_constraints = torch.mean(F.relu(torch.abs(y_pred) - 10.0))  # Max 10°C bias
        
        return spatial_smoothness + temporal_consistency + physical_constraints
    
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
        x = batch['input']  # Shape: (batch_size, seq_len, input_dim)
        y = batch['target']  # Shape: (batch_size, seq_len, output_dim)
        
        # Forward pass
        y_pred, _ = self(x)
        
        # Calculate losses
        mse_loss = F.mse_loss(y_pred, y)
        physics_loss = self.calculate_physics_loss(y_pred, y)
        
        # Total loss
        total_loss = mse_loss + self.physics_weight * physics_loss
        
        # Log metrics
        self.log('train/mse_loss', mse_loss)
        self.log('train/physics_loss', physics_loss)
        self.log('train/total_loss', total_loss)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.
        
        Args:
            batch (Dict[str, torch.Tensor]): Batch of data
            batch_idx (int): Batch index
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of metrics
        """
        # Extract data from batch
        x = batch['input']
        y = batch['target']
        
        # Forward pass
        y_pred, _ = self(x)
        
        # Calculate metrics
        mse_loss = F.mse_loss(y_pred, y)
        mae = F.l1_loss(y_pred, y)
        rmse = torch.sqrt(mse_loss)
        
        # Log metrics
        self.log('val/mse_loss', mse_loss)
        self.log('val/mae', mae)
        self.log('val/rmse', rmse)
        
        return {
            'val/mse_loss': mse_loss,
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
            Dict[str, torch.Tensor]: Dictionary of metrics
        """
        # Extract data from batch
        x = batch['input']
        y = batch['target']
        
        # Forward pass
        y_pred, _ = self(x)
        
        # Calculate metrics
        mse_loss = F.mse_loss(y_pred, y)
        mae = F.l1_loss(y_pred, y)
        rmse = torch.sqrt(mse_loss)
        
        # Calculate calibration metrics
        calibration_metrics = self.calculate_calibration_metrics(y_pred, y)
        
        # Log metrics
        self.log('test/mse_loss', mse_loss)
        self.log('test/mae', mae)
        self.log('test/rmse', rmse)
        for metric_name, metric_value in calibration_metrics.items():
            self.log(f'test/{metric_name}', metric_value)
        
        return {
            'test/mse_loss': mse_loss,
            'test/mae': mae,
            'test/rmse': rmse,
            **calibration_metrics
        } 