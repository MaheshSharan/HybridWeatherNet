import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class AttentionModule(nn.Module):
    """
    Attention module for feature fusion in bias correction.
    
    This module uses multi-head attention to combine temporal and spatial features.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout_rate: float = 0.2
    ):
        """
        Initialize the attention module.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layers
            num_heads (int): Number of attention heads
            dropout_rate (float): Dropout rate for regularization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        # Linear projections for query, key, value
        self.q_proj = nn.Linear(input_dim, hidden_dim)
        self.k_proj = nn.Linear(input_dim, hidden_dim)
        self.v_proj = nn.Linear(input_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize the parameters of the attention module."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the attention module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, seq_len, seq_len)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Output tensor of shape (batch_size, seq_len, hidden_dim)
                - Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        # Apply layer normalization
        x = self.layer_norm1(x)
        
        # Project queries, keys, and values
        q = self.q_proj(x)  # (batch_size, seq_len, hidden_dim)
        k = self.k_proj(x)  # (batch_size, seq_len, hidden_dim)
        v = self.v_proj(x)  # (batch_size, seq_len, hidden_dim)
        
        # Reshape for multi-head attention
        batch_size, seq_len, _ = x.shape
        q = q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(context)
        
        # Apply layer normalization and residual connection
        output = self.layer_norm2(output + x)
        
        return output, attn_weights
    
    def get_output_dim(self) -> int:
        """
        Get the output dimension of the attention module.
        
        Returns:
            int: Output dimension
        """
        return self.hidden_dim 