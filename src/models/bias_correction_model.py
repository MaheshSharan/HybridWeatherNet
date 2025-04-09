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
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
            physics_weight=physics_weight
        )
        
        self.lstm = LSTMModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.gnn = GraphModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )
        
        combined_dim = lstm_output_dim + hidden_dim
        
        print(f"Model init - LSTM dim: {lstm_output_dim}, GNN dim: {hidden_dim}, Combined: {combined_dim}")
        
        self.attention = AttentionModule(
            input_dim=combined_dim,
            hidden_dim=hidden_dim,
            num_heads=8,
            dropout=dropout_rate
        )
        
        self.output_proj = nn.Linear(combined_dim, output_dim)
        
        self.num_mc_samples = num_mc_samples
        self.mc_dropout = nn.Dropout(p=dropout_rate)
        
        self.save_hyperparameters()
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        lstm_out, _ = self.lstm(x)
        
        if edge_index is not None:
            gnn_out = self.gnn(x.view(-1, x.size(-1)), edge_index, edge_attr)
            gnn_out = gnn_out.view(x.size(0), x.size(1), -1)
        else:
            gnn_out = torch.zeros(x.size(0), x.size(1), self.hparams.hidden_dim, device=x.device, dtype=x.dtype)
        
        combined = torch.cat([lstm_out, gnn_out], dim=-1)
        attended, attention_weights = self.attention(combined)
        attended_last = attended[:, -1, :]
        output = self.output_proj(attended_last)
        
        if return_uncertainty:
            mc_outputs = []
            for _ in range(self.num_mc_samples):
                mc_out = self.mc_dropout(attended_last)
                mc_out = self.output_proj(mc_out)
                mc_outputs.append(mc_out)
            
            mc_outputs = torch.stack(mc_outputs, dim=0)
            mean = mc_outputs.mean(dim=0)
            variance = mc_outputs.var(dim=0)
            return mean, attention_weights, variance
        
        dummy_uncertainty = torch.zeros_like(output)
        return output, attention_weights, dummy_uncertainty
    
    def estimate_uncertainty(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.train()
        with torch.no_grad():
            mean, _, variance = self(x, edge_index, edge_attr, return_uncertainty=True)
        self.eval()
        return mean, variance
    
    def calculate_physics_loss(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        if torch.isnan(y_pred).any():
            y_pred = torch.nan_to_num(y_pred, nan=0.0)
        
        spatial_smoothness = torch.tensor(0.0, device=y_pred.device)
        temporal_consistency = torch.tensor(0.0, device=y_pred.device)
        physical_constraints = torch.tensor(0.0, device=y_pred.device)
        
        try:
            spatial_diff = y_pred[1:] - y_pred[:-1]
            spatial_smoothness = torch.mean(torch.abs(spatial_diff))
        except Exception:
            pass
        
        try:
            temperature = x[:, :, 0]
            humidity = x[:, :, 1]
            wind_speed = x[:, :, 2]
            humidity_factor = torch.exp(-humidity[:, -1])
            wind_factor = torch.exp(-wind_speed[:, -1])
            physical_factor = humidity_factor * wind_factor
            expected_bias_magnitude = 5.0 * (1.0 - physical_factor)
            actual_bias_magnitude = torch.abs(y_pred)
            temporal_consistency = torch.mean(torch.abs(actual_bias_magnitude - expected_bias_magnitude))
        except Exception:
            pass
        
        try:
            physical_constraints = torch.mean(F.relu(torch.abs(y_pred) - 10.0))
        except Exception:
            pass
        
        if torch.isnan(spatial_smoothness):
            spatial_smoothness = torch.tensor(0.0, device=y_pred.device)
        if torch.isnan(temporal_consistency):
            temporal_consistency = torch.tensor(0.0, device=y_pred.device)
        if torch.isnan(physical_constraints):
            physical_constraints = torch.tensor(0.0, device=y_pred.device)
        
        total_physics_loss = 2.0 * spatial_smoothness + 1.5 * temporal_consistency + physical_constraints
        if torch.isnan(total_physics_loss):
            total_physics_loss = torch.tensor(0.0, device=y_pred.device)
        
        return total_physics_loss
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch['input']
        y = batch['target'].squeeze(1)
        
        y_pred, _, _ = self(x)
        
        if torch.isnan(y_pred).any():
            print(f"NaN in predictions at batch {batch_idx}")
            y_pred = torch.nan_to_num(y_pred, nan=0.0)
        
        try:
            mse_loss = F.mse_loss(y_pred, y)
            mae_loss = F.l1_loss(y_pred, y)
            rmse_loss = torch.sqrt(mse_loss)
            physics_loss = self.calculate_physics_loss(x, y_pred, y)
            total_loss = mse_loss + self.physics_weight * physics_loss
            
            if torch.isnan(total_loss):
                total_loss = mse_loss + torch.tensor(0.0, device=mse_loss.device)
        except Exception as e:
            print(f"Error in loss calculation: {str(e)}")
            total_loss = torch.tensor(1.0, device=x.device)
            mse_loss = total_loss
            mae_loss = torch.tensor(0.0, device=x.device)
            rmse_loss = torch.tensor(0.0, device=x.device)
            physics_loss = torch.tensor(0.0, device=x.device)
        
        self.log('train/loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/mse_loss', mse_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/mae_loss', mae_loss, on_step=False, on_epoch=True)
        self.log('train/rmse_loss', rmse_loss, on_step=False, on_epoch=True)
        self.log('train/physics_loss', physics_loss, on_step=False, on_epoch=True)
        self.log('train/learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        x = batch['input']
        y = batch['target'].squeeze(1)
        
        y_pred, _, uncertainty = self(x)
        
        mse_loss = F.mse_loss(y_pred, y)
        mae = F.l1_loss(y_pred, y)
        rmse = torch.sqrt(mse_loss)
        physics_loss = self.calculate_physics_loss(x, y_pred, y)
        total_loss = mse_loss + self.physics_weight * physics_loss
        
        self.log('val/loss', total_loss, prog_bar=True)
        self.log('val/mse_loss', mse_loss, prog_bar=True)
        self.log('val/mae', mae, prog_bar=True)
        self.log('val/rmse', rmse, prog_bar=True)
        self.log('val/physics_loss', physics_loss, prog_bar=True)
        
        return {
            'val/loss': total_loss,
            'val/mse_loss': mse_loss,
            'val/mae': mae,
            'val/rmse': rmse,
            'val/physics_loss': physics_loss
        }
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        x = batch['input']
        y = batch['target'].squeeze(1)
        
        y_pred, attention_weights, uncertainty = self(x)
        
        mse_loss = F.mse_loss(y_pred, y)
        mae = F.l1_loss(y_pred, y)
        rmse = torch.sqrt(mse_loss)
        physics_loss = self.calculate_physics_loss(x, y_pred, y)
        total_loss = mse_loss + self.physics_weight * physics_loss
        
        self.log('test/mse_loss', mse_loss, prog_bar=True)
        self.log('test/mae', mae, prog_bar=True)
        self.log('test/rmse', rmse, prog_bar=True)
        self.log('test/physics_loss', physics_loss, prog_bar=True)
        self.log('test/total_loss', total_loss, prog_bar=True)
        
        return {
            'test/mse_loss': mse_loss,
            'test/mae': mae,
            'test/rmse': rmse,
            'test/physics_loss': physics_loss,
            'test/total_loss': total_loss
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch"
            }
        }