import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        assert input_dim % num_heads == 0, "Input dimension must be divisible by num_heads"
        self.head_dim = input_dim // num_heads
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.output_proj = nn.Linear(input_dim, input_dim)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, input_dim = x.size()
        assert input_dim == self.input_dim, f"Expected input_dim {self.input_dim}, got {input_dim}"
        
        q = self.query(x)  # [batch_size, seq_len, input_dim]
        k = self.key(x)    # [batch_size, seq_len, input_dim]
        v = self.value(x)  # [batch_size, seq_len, input_dim]
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.input_dim)
        
        output = self.output_proj(attn_output)
        
        if output.size(-1) != self.input_dim:
            print(f"Warning: Output dimension {output.size(-1)} doesn't match expected dimension {self.input_dim}")
        
        return output, attn_weights