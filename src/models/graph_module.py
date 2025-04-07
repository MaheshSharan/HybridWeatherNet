import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import torch_geometric.nn as gnn

class GraphModule(nn.Module):
    """
    Graph Neural Network module for spatial pattern learning in bias correction.
    
    This module processes spatial relationships between weather stations using
    a Graph Neural Network architecture.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout_rate: float = 0.2,
        edge_dim: int = 1  # Dimension of edge features (e.g., distance)
    ):
        """
        Initialize the Graph Neural Network module.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layers
            num_layers (int): Number of GNN layers
            dropout_rate (float): Dropout rate for regularization
            edge_dim (int): Dimension of edge features
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.edge_dim = edge_dim
        
        # Graph Neural Network layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(
            gnn.GATConv(
                in_channels=input_dim,
                out_channels=hidden_dim,
                heads=4,
                dropout=dropout_rate,
                edge_dim=edge_dim
            )
        )
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(
                gnn.GATConv(
                    in_channels=hidden_dim * 4,  # 4 heads from previous layer
                    out_channels=hidden_dim,
                    heads=4,
                    dropout=dropout_rate,
                    edge_dim=edge_dim
                )
            )
        
        # Final layer
        self.convs.append(
            gnn.GATConv(
                in_channels=hidden_dim * 4,
                out_channels=hidden_dim,
                heads=1,
                dropout=dropout_rate,
                edge_dim=edge_dim
            )
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the Graph Neural Network.
        
        Args:
            x (torch.Tensor): Node features of shape (num_nodes, input_dim)
            edge_index (torch.Tensor): Graph connectivity of shape (2, num_edges)
            edge_attr (Optional[torch.Tensor]): Edge features of shape (num_edges, edge_dim)
            
        Returns:
            torch.Tensor: Node embeddings of shape (num_nodes, hidden_dim)
        """
        # Apply GNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            if i < len(self.convs) - 1:  # Don't apply activation after last layer
                x = F.elu(x)
                x = self.dropout(x)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        return x
    
    def get_output_dim(self) -> int:
        """
        Get the output dimension of the Graph Neural Network.
        
        Returns:
            int: Output dimension
        """
        return self.hidden_dim 