import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class LSTMModule(nn.Module):
    """
    LSTM module for temporal pattern learning in bias correction.
    
    This module processes sequential data to capture temporal patterns in weather forecasts.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout_rate: float = 0.2,
        bidirectional: bool = True
    ):
        """
        Initialize the LSTM module.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layers
            num_layers (int): Number of LSTM layers
            dropout_rate (float): Dropout rate for regularization
            bidirectional (bool): Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        
        # Calculate output dimension based on bidirectional setting
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.output_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the LSTM module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            hidden (Optional[Tuple[torch.Tensor, torch.Tensor]]): Initial hidden state
            
        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: 
                - Output tensor of shape (batch_size, seq_len, output_dim)
                - Final hidden state
        """
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Apply layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        return lstm_out, hidden
    
    def get_output_dim(self) -> int:
        """
        Get the output dimension of the LSTM module.
        
        Returns:
            int: Output dimension
        """
        return self.output_dim 