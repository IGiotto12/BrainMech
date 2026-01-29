"""
Task Context Encoder
Encodes task vectors and high-level prompts for control
"""

import torch
import torch.nn as nn


class TaskContextEncoder(nn.Module):
    """
    Task Context Encoder
    Encodes task/design vectors and high-level prompts
    Provides context for control signal generation
    """
    
    def __init__(
        self,
        d_task: int,
        d_context: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            d_task: Task vector dimension
            d_context: Context embedding dimension
            n_layers: Number of encoding layers
            dropout: Dropout rate
        """
        super().__init__()
        self.d_task = d_task
        self.d_context = d_context
        
        layers = []
        for i in range(n_layers):
            in_dim = d_task if i == 0 else d_context
            out_dim = d_context
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.LayerNorm(out_dim),
                nn.Dropout(dropout)
            ])
        
        self.encoder = nn.Sequential(*layers)
        
        self.output_proj = nn.Linear(d_context, d_context)
        
    def forward(self, task_vectors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            task_vectors: [B, T, d_task] - Task/design vectors
        Returns:
            context_embedding: [B, T, d_context] - Context embeddings
        """
        context = self.encoder(task_vectors)
        context = self.output_proj(context)
        return context


if __name__ == "__main__":
    # Test Task Context Encoder
    B, T, d_task = 4, 100, 32
    d_context = 64
    
    task_encoder = TaskContextEncoder(
        d_task=d_task,
        d_context=d_context,
        n_layers=2
    )
    
    task_vec = torch.randn(B, T, d_task)
    context = task_encoder(task_vec)
    
    print(f"Input shape: {task_vec.shape}")
    print(f"Context embedding shape: {context.shape}")
    print(f"Parameters: {sum(p.numel() for p in task_encoder.parameters()):,}")

