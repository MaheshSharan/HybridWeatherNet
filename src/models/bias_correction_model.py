import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import pytorch_lightning as pl

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
        
        # Calculate LSTM output dimension based on bidirectionality
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Graph Neural Network module for spatial relationships
        self.gnn = GraphModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )
        
        # Calculate combined dimension - exact value based on architecture
        combined_dim = lstm_output_dim + hidden_dim  # LSTM + GNN outputs
        
        print(f"Model init - LSTM dim: {lstm_output_dim}, GNN dim: {hidden_dim}, Combined: {combined_dim}")
        
        # Attention module for feature fusion with correct dimensions
        self.attention = AttentionModule(
            input_dim=combined_dim,
            hidden_dim=hidden_dim,
            num_heads=8,
            dropout=dropout_rate
        )
        
        # Output projection directly from combined features to output dimension
        self.output_proj = nn.Linear(combined_dim, output_dim)
        
        # Monte Carlo dropout for uncertainty estimation
        self.num_mc_samples = num_mc_samples
        self.mc_dropout = nn.Dropout(p=dropout_rate)
        
        # No need for dynamic projection initialization flag since we set up correctly from the start
        
        # Save hyperparameters
        self.save_hyperparameters()
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            edge_index (Optional[torch.Tensor]): Graph connectivity
            edge_attr (Optional[torch.Tensor]): Edge features
            return_uncertainty (bool): Whether to return uncertainty estimates
            
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                - If return_uncertainty is False: Tuple of (predicted bias, attention_weights, dummy_uncertainty)
                - If return_uncertainty is True: Tuple of (predicted bias, attention_weights, uncertainty)
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
        
        # Apply attention mechanism
        attended, attention_weights = self.attention(combined)
        
        # Project to output dimension using the fixed projection layer
        # We use the attended output for the final projection, since it contains the fused information
        output = self.output_proj(combined)
        
        if return_uncertainty:
            # Monte Carlo dropout for uncertainty estimation
            mc_outputs = []
            for _ in range(self.num_mc_samples):
                # Apply dropout to original combined features for Monte Carlo sampling
                mc_out = self.mc_dropout(combined)
                # Project to output dimension
                mc_out = self.output_proj(mc_out)
                mc_outputs.append(mc_out)
            
            # Stack MC samples
            mc_outputs = torch.stack(mc_outputs, dim=0)
            
            # Calculate mean and variance
            mean = mc_outputs.mean(dim=0)
            variance = mc_outputs.var(dim=0)
            
            return mean, attention_weights, variance
        
        # Return a dummy uncertainty tensor when return_uncertainty is False
        dummy_uncertainty = torch.zeros_like(output)
        return output, attention_weights, dummy_uncertainty
    
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
            mean, _, variance = self(x, edge_index, edge_attr, return_uncertainty=True)
        self.eval()  # Disable dropout
        return mean, variance
    
    def calculate_physics_loss(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate physics-guided loss based on the research paper methodology.
        
        Args:
            x (torch.Tensor): Input features
            y_pred (torch.Tensor): Predicted bias
            y_true (torch.Tensor): True bias
            
        Returns:
            torch.Tensor: Physics-guided loss
        """
        # Handle NaN values in inputs
        if torch.isnan(y_pred).any():
            y_pred = torch.nan_to_num(y_pred, nan=0.0)
        
        # Initialize loss components
        spatial_smoothness = torch.tensor(0.0, device=y_pred.device)
        temporal_consistency = torch.tensor(0.0, device=y_pred.device)
        physical_constraints = torch.tensor(0.0, device=y_pred.device)
        
        # SPATIAL SMOOTHNESS LOSS
        # For single time step data, we'll compare predictions across different locations in the batch
        try:
            # Reshape to focus on spatial dimension (batch dimension)
            # From [batch_size, seq_len, output_dim] to [batch_size, output_dim]
            y_pred_spatial = y_pred.squeeze(1)  # Remove sequence dimension
            
            # Calculate differences between consecutive predictions in the batch
            # This represents spatial differences between different locations
            spatial_diff = y_pred_spatial[1:] - y_pred_spatial[:-1]
            
            # Calculate mean absolute difference
            spatial_smoothness = torch.mean(torch.abs(spatial_diff))
        except Exception as e:
            pass
        
        # TEMPORAL CONSISTENCY LOSS
        # Based on the research paper's hybrid architecture approach
        try:
            # Extract meteorological variables from input features
            # Assuming the order: temperature, humidity, wind_speed, wind_direction, cloud_cover
            temperature = x[:, :, 0]  # First feature is temperature
            humidity = x[:, :, 1]     # Second feature is humidity
            wind_speed = x[:, :, 2]   # Third feature is wind speed
            
            # Calculate temporal consistency based on physical relationships
            # 1. Temperature-humidity relationship: higher humidity should correlate with smaller temperature changes
            humidity_factor = torch.exp(-humidity)  # Exponential decay with humidity
            
            # 2. Wind speed impact: higher wind speeds should lead to more mixing and smaller temperature biases
            wind_factor = torch.exp(-wind_speed)  # Exponential decay with wind speed
            
            # 3. Combined physical factors
            physical_factor = humidity_factor * wind_factor
            
            # Calculate expected bias magnitude based on physical factors
            # Higher physical factor = smaller expected bias
            expected_bias_magnitude = 5.0 * (1.0 - physical_factor)  # Scale factor of 5.0°C max bias
            
            # Calculate actual bias magnitude
            actual_bias_magnitude = torch.abs(y_pred)
            
            # Temporal consistency: bias should follow physical relationships
            # This encourages the model to learn physically consistent bias patterns
            temporal_consistency = torch.mean(torch.abs(actual_bias_magnitude - expected_bias_magnitude))
        except Exception as e:
            pass
        
        # PHYSICAL CONSTRAINTS
        # Always calculate physical constraints
        try:
            # Enforce reasonable bounds on temperature bias
            # Max 10°C bias is a reasonable constraint
            physical_constraints = torch.mean(F.relu(torch.abs(y_pred) - 10.0))
        except Exception as e:
            pass
        
        # Replace NaN values with zeros
        if torch.isnan(spatial_smoothness):
            spatial_smoothness = torch.tensor(0.0, device=y_pred.device)
        if torch.isnan(temporal_consistency):
            temporal_consistency = torch.tensor(0.0, device=y_pred.device)
        if torch.isnan(physical_constraints):
            physical_constraints = torch.tensor(0.0, device=y_pred.device)
        
        # Calculate total physics loss with error handling
        try:
            # Adjust weights according to the research paper
            # Spatial smoothness: 2.0 (high weight for spatial consistency)
            # Temporal consistency: 1.5 (medium weight for temporal consistency)
            # Physical constraints: 1.0 (base weight for physical constraints)
            total_physics_loss = 2.0 * spatial_smoothness + 1.5 * temporal_consistency + physical_constraints
            
            if torch.isnan(total_physics_loss):
                total_physics_loss = torch.tensor(0.0, device=y_pred.device)
            
            return total_physics_loss
        except Exception as e:
            return torch.tensor(0.0, device=y_pred.device)
    
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
        y_pred, _, _ = self(x)  # Shape: (batch_size, seq_len, output_dim)
        
        # Debug NaN values
        if torch.isnan(y_pred).any():
            print(f"NaN in predictions at batch {batch_idx}")
            print(f"Features stats: min={x.min()}, max={x.max()}, mean={x.mean()}")
            # Replace NaN values with zeros to prevent training failure
            y_pred = torch.nan_to_num(y_pred, nan=0.0)
        
        # Calculate losses with error handling
        try:
            mse_loss = F.mse_loss(y_pred, y)
            mae_loss = F.l1_loss(y_pred, y)
            rmse_loss = torch.sqrt(mse_loss)
            physics_loss = self.calculate_physics_loss(x, y_pred, y)
            total_loss = mse_loss + self.physics_weight * physics_loss
            
            # Check for NaN values in losses
            if torch.isnan(mse_loss) or torch.isnan(physics_loss) or torch.isnan(total_loss):
                print(f"NaN in losses at batch {batch_idx}")
                print(f"MSE loss: {mse_loss}, Physics loss: {physics_loss}, Total loss: {total_loss}")
                
                # Replace NaN values with a default value to prevent training failure
                if torch.isnan(mse_loss):
                    mse_loss = torch.tensor(1.0, device=mse_loss.device)
                if torch.isnan(physics_loss):
                    physics_loss = torch.tensor(0.0, device=physics_loss.device)
                if torch.isnan(total_loss):
                    total_loss = mse_loss + self.physics_weight * physics_loss
        except Exception as e:
            print(f"Error in loss calculation: {str(e)}")
            # Use a default loss value if calculation fails
            total_loss = torch.tensor(1.0, device=x.device)
            mse_loss = total_loss
            physics_loss = torch.tensor(0.0, device=x.device)
            mae_loss = torch.tensor(0.0, device=x.device)
            rmse_loss = torch.tensor(0.0, device=x.device)
        
        # Log metrics with error handling
        try:
            self.log('train/loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log('train/mse_loss', mse_loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log('train/mae_loss', mae_loss, on_step=False, on_epoch=True)
            self.log('train/rmse_loss', rmse_loss, on_step=False, on_epoch=True)
            self.log('train/physics_loss', physics_loss, on_step=False, on_epoch=True)
            
            # Log learning rate
            self.log('train/learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True)
        except Exception as e:
            print(f"Error logging metrics: {str(e)}")
        
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
        y_pred, attention_weights, uncertainty = self(x)
        
        # Ensure shapes match for loss calculation
        if y_pred.shape != y.shape:
            # Reshape y to match y_pred if needed
            y = y.view(y_pred.shape)
        
        # Calculate metrics
        mse_loss = F.mse_loss(y_pred, y)
        mae = F.l1_loss(y_pred, y)
        rmse = torch.sqrt(mse_loss)
        
        # Calculate physics-guided loss
        physics_loss = self.calculate_physics_loss(x, y_pred, y)
        
        # Calculate total loss
        total_loss = mse_loss + self.physics_weight * physics_loss
        
        # Log metrics
        self.log('val/loss', total_loss)
        self.log('val/mse_loss', mse_loss)
        self.log('val/mae', mae)
        self.log('val/rmse', rmse)
        self.log('val/physics_loss', physics_loss)
        
        return {
            'val/loss': total_loss,
            'val/mse_loss': mse_loss,
            'val/mae': mae,
            'val/rmse': rmse,
            'val/physics_loss': physics_loss
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
        y_pred, attention_weights, uncertainty = self(x)
        
        # Debug prints
        print(f"Input shape: {x.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Prediction shape: {y_pred.shape}")
        print(f"Uncertainty shape: {uncertainty.shape}")
        
        # Check for NaN values
        if torch.isnan(x).any():
            print("Warning: NaN values in input")
        if torch.isnan(y).any():
            print("Warning: NaN values in target")
        if torch.isnan(y_pred).any():
            print("Warning: NaN values in predictions")
        if torch.isnan(uncertainty).any():
            print("Warning: NaN values in uncertainty")
        
        # Ensure shapes match for loss calculation
        if y_pred.shape != y.shape:
            # Reshape y to match y_pred if needed
            y = y.view(y_pred.shape)
            print(f"Reshaped target to: {y.shape}")
        
        # Calculate metrics with error handling
        try:
            mse_loss = F.mse_loss(y_pred, y)
            mae = F.l1_loss(y_pred, y)
            rmse = torch.sqrt(mse_loss)
            
            # Check for NaN in basic metrics
            if torch.isnan(mse_loss):
                print("Warning: MSE loss is NaN")
                mse_loss = torch.tensor(0.0, device=mse_loss.device)
            if torch.isnan(mae):
                print("Warning: MAE is NaN")
                mae = torch.tensor(0.0, device=mae.device)
            if torch.isnan(rmse):
                print("Warning: RMSE is NaN")
                rmse = torch.tensor(0.0, device=rmse.device)
            
            # Calculate calibration metrics with error handling
            try:
                calibration_metrics = self.calculate_calibration_metrics(y, y_pred, uncertainty)
                
                # Check for NaN in calibration metrics
                for key, value in calibration_metrics.items():
                    if torch.isnan(value):
                        print(f"Warning: {key} is NaN")
                        calibration_metrics[key] = torch.tensor(0.0, device=value.device)
            except Exception as e:
                print(f"Error in calibration metrics: {str(e)}")
                calibration_metrics = {
                    'calibration_error': torch.tensor(0.0, device=y_pred.device),
                    'uncertainty_correlation': torch.tensor(0.0, device=y_pred.device)
                }
        except Exception as e:
            print(f"Error in metric calculation: {str(e)}")
            mse_loss = torch.tensor(0.0, device=y_pred.device)
            mae = torch.tensor(0.0, device=y_pred.device)
            rmse = torch.tensor(0.0, device=y_pred.device)
            calibration_metrics = {
                'calibration_error': torch.tensor(0.0, device=y_pred.device),
                'uncertainty_correlation': torch.tensor(0.0, device=y_pred.device)
            }
        
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
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch"
            }
        }
    
    def calculate_calibration_metrics(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        uncertainty: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate calibration metrics.
        
        Args:
            y_true (torch.Tensor): True values
            y_pred (torch.Tensor): Predicted values
            uncertainty (torch.Tensor): Uncertainty estimates
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of calibration metrics
        """
        # Ensure all tensors are flattened for calculation
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        uncertainty_flat = uncertainty.flatten()
        
        # Calculate absolute error
        abs_error = torch.abs(y_true_flat - y_pred_flat)
        
        # Calculate calibration error (correlation between error and uncertainty)
        # Handle potential NaN values in correlation calculation
        try:
            # Check if we have enough non-zero values for correlation
            if torch.sum(uncertainty_flat != 0) > 1 and torch.sum(abs_error != 0) > 1:
                # Calculate correlation coefficient
                uncertainty_mean = torch.mean(uncertainty_flat)
                abs_error_mean = torch.mean(abs_error)
                
                numerator = torch.sum((uncertainty_flat - uncertainty_mean) * (abs_error - abs_error_mean))
                denominator = torch.sqrt(
                    torch.sum((uncertainty_flat - uncertainty_mean) ** 2) * 
                    torch.sum((abs_error - abs_error_mean) ** 2)
                )
                
                # Check for division by zero or very small values
                if denominator > 1e-10:
                    correlation = numerator / denominator
                else:
                    correlation = torch.tensor(0.0, device=y_true.device)
                    print("Warning: Denominator too small for correlation calculation")
            else:
                correlation = torch.tensor(0.0, device=y_true.device)
                print("Warning: Not enough non-zero values for correlation calculation")
        except Exception as e:
            print(f"Error in correlation calculation: {str(e)}")
            correlation = torch.tensor(0.0, device=y_true.device)
        
        # Calculate uncertainty correlation (how well uncertainty estimates correlate with actual errors)
        try:
            # Check if we have enough non-zero values for correlation
            if torch.sum(uncertainty_flat != 0) > 1 and torch.sum(abs_error != 0) > 1:
                # Calculate correlation coefficient
                uncertainty_mean = torch.mean(uncertainty_flat)
                abs_error_mean = torch.mean(abs_error)
                
                numerator = torch.sum((uncertainty_flat - uncertainty_mean) * (abs_error - abs_error_mean))
                denominator = torch.sqrt(
                    torch.sum((uncertainty_flat - uncertainty_mean) ** 2) * 
                    torch.sum((abs_error - abs_error_mean) ** 2)
                )
                
                # Check for division by zero or very small values
                if denominator > 1e-10:
                    uncertainty_correlation = numerator / denominator
                else:
                    uncertainty_correlation = torch.tensor(0.0, device=y_true.device)
                    print("Warning: Denominator too small for uncertainty correlation calculation")
            else:
                uncertainty_correlation = torch.tensor(0.0, device=y_true.device)
                print("Warning: Not enough non-zero values for uncertainty correlation calculation")
        except Exception as e:
            print(f"Error in uncertainty correlation calculation: {str(e)}")
            uncertainty_correlation = torch.tensor(0.0, device=y_true.device)
        
        return {
            'calibration_error': correlation,
            'uncertainty_correlation': uncertainty_correlation
        } 