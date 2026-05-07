"""
True Graph-Based Spatial Encoder with Message Passing
Implements ChebNet (spectral) or GAT (spatial) with DropEdge regularization

Contribution 1: Real graph structure modeling
- Message passing over FC/SC adjacency
- DropEdge for stochastic graph regularization
- Graph-spectral or graph-attention layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class ChebConv(nn.Module):
    """
    Chebyshev Spectral Graph Convolution
    More efficient than full spectral convolution
    """
    
    def __init__(self, in_features: int, out_features: int, K: int = 3):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            K: Chebyshev polynomial order
        """
        super().__init__()
        self.K = K
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable weights for each polynomial order
        self.weight = nn.Parameter(torch.Tensor(K, in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, d_in] or [B, N, d_in] Node features
            laplacian: [N, N] Normalized graph Laplacian
        
        Returns:
            out: [B, T, N, d_out] or [B, N, d_out]
        """
        # Handle both 3D and 4D tensors
        if x.dim() == 3:
            B, N, d = x.shape
            x = x.unsqueeze(1)  # [B, 1, N, d]
            squeeze_output = True
        else:
            B, T, N, d = x.shape
            squeeze_output = False
        
        # Flatten B, T dimensions for processing
        x_flat = x.reshape(-1, N, d)  # [B*T, N, d]
        
        # Compute Chebyshev polynomials of Laplacian
        # Clear cache before large tensor operations to reduce fragmentation
        if x_flat.size(0) > 10000:  # Only for large batches
            torch.cuda.empty_cache()
        
        Tx_0 = x_flat  # T_0 = I
        Tx_1 = torch.bmm(laplacian.unsqueeze(0).expand(x_flat.size(0), -1, -1), x_flat)  # T_1 = L
        
        out = torch.matmul(Tx_0, self.weight[0])
        if self.K > 1:
            out += torch.matmul(Tx_1, self.weight[1])
        
        # Recurrence: T_k = 2L·T_{k-1} - T_{k-2}
        for k in range(2, self.K):
            Tx_2 = 2 * torch.bmm(laplacian.unsqueeze(0).expand(x_flat.size(0), -1, -1), Tx_1) - Tx_0
            out += torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2
        
        out = out + self.bias
        
        # Reshape back
        if squeeze_output:
            out = out.reshape(B, N, self.out_features)
        else:
            out = out.reshape(B, T, N, self.out_features)
        
        return out


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT) with edge-wise attention
    """
    
    def __init__(self, in_features: int, out_features: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.d_k = out_features // n_heads
        
        assert out_features % n_heads == 0, "out_features must be divisible by n_heads"
        
        # Linear transformations
        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Parameter(torch.Tensor(n_heads, 2 * self.d_k, 1))
        
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, d_in] Node features (single timepoint)
            adj: [N, N] Adjacency matrix (weighted, symmetric)
        
        Returns:
            out: [B, N, d_out] Updated node features
        """
        B, N, _ = x.shape
        
        # Linear transformation
        h = self.W(x)  # [B, N, d_out]
        h = h.view(B, N, self.n_heads, self.d_k)  # [B, N, heads, d_k]
        
        # Prepare for attention
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1, -1)  # [B, N, N, heads, d_k]
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1, -1)  # [B, N, N, heads, d_k]
        
        # Concatenate for attention
        h_cat = torch.cat([h_i, h_j], dim=-1)  # [B, N, N, heads, 2*d_k]
        
        # Compute attention scores
        e = self.leakyrelu(torch.matmul(h_cat, self.a).squeeze(-1))  # [B, N, N, heads]
        
        # Mask attention with adjacency matrix
        mask = (adj == 0).unsqueeze(0).unsqueeze(-1)  # [1, N, N, 1]
        e = e.masked_fill(mask, -1e9)
        
        # Softmax over neighbors
        alpha = F.softmax(e, dim=2)  # [B, N, N, heads]
        alpha = self.dropout(alpha)
        
        # Aggregate
        h = h.permute(0, 2, 1, 3)  # [B, heads, N, d_k]
        alpha = alpha.permute(0, 3, 1, 2)  # [B, heads, N, N]
        out = torch.matmul(alpha, h.permute(0, 1, 3, 2))  # [B, heads, N, d_k]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, -1)  # [B, N, d_out]
        
        return out


def compute_normalized_laplacian(adj: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    """
    Compute normalized graph Laplacian: L = I - D^{-1/2} A D^{-1/2}
    
    Args:
        adj: [N, N] Adjacency matrix
        add_self_loops: Whether to add self-connections
    
    Returns:
        L_norm: [N, N] Normalized Laplacian
    """
    N = adj.size(0)
    
    if add_self_loops:
        adj = adj + torch.eye(N, device=adj.device)
    
    # Degree matrix
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    
    # D^{-1/2} A D^{-1/2}
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    L_norm = torch.eye(N, device=adj.device) - torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
    
    return L_norm


class DropEdge(nn.Module):
    """
    DropEdge: Stochastic edge dropping for graph regularization
    """
    
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p
    
    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adj: [N, N] Adjacency matrix
        
        Returns:
            adj_dropped: [N, N] Adjacency with randomly dropped edges
        """
        if not self.training or self.p == 0:
            return adj
        
        # Create edge mask (keep diagonal)
        mask = torch.rand_like(adj) > self.p
        mask = mask | torch.eye(adj.size(0), device=adj.device, dtype=torch.bool)
        
        # Make symmetric
        mask = mask & mask.T
        
        # Apply mask and renormalize
        adj_dropped = adj * mask.float()
        
        # Renormalize to maintain expected sum
        adj_dropped = adj_dropped / (1 - self.p + 1e-8)
        
        return adj_dropped


class GraphSpatialEncoderReal(nn.Module):
    """
    True Graph-Based Spatial Encoder with Message Passing
    
    Uses either:
    - ChebNet (spectral graph convolution)
    - GAT (graph attention)
    
    With DropEdge for stochastic regularization
    """
    
    def __init__(
        self,
        n_roi: int,
        d_latent: int,
        n_layers: int = 3,
        graph_type: str = 'cheb',  # 'cheb' or 'gat'
        cheb_k: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        drop_edge_p: float = 0.1,
        fc_matrix: Optional[torch.Tensor] = None,
        use_temporal_encoding: bool = True
    ):
        """
        Args:
            n_roi: Number of ROIs (graph nodes)
            d_latent: Latent dimension
            n_layers: Number of graph conv layers
            graph_type: 'cheb' (spectral) or 'gat' (attention)
            cheb_k: Chebyshev polynomial order
            n_heads: Number of attention heads (for GAT)
            dropout: Dropout rate
            drop_edge_p: DropEdge probability
            fc_matrix: [N, N] Functional connectivity adjacency
            use_temporal_encoding: Whether to add temporal positional encoding
        """
        super().__init__()
        self.n_roi = n_roi
        self.d_latent = d_latent
        self.n_layers = n_layers
        self.graph_type = graph_type
        self.use_temporal_encoding = use_temporal_encoding
        
        # Input projection: ROI signal → node features
        self.input_proj = nn.Linear(1, d_latent)  # Each ROI is a scalar at each time
        
        # Graph structure
        if fc_matrix is not None:
            # Drop negative connections, then normalize FC matrix to [0, 1]
            # Only keep positive functional connectivity (excitatory connections)
            fc_positive = torch.clamp(fc_matrix, min=0.0)  # Set negative to 0
            fc_norm = fc_positive / (fc_positive.max() + 1e-8)  # Normalize to [0, 1]
            self.register_buffer('adj_matrix', fc_norm)
            
            # Compute Laplacian for ChebNet
            if graph_type == 'cheb':
                self.register_buffer('laplacian', compute_normalized_laplacian(fc_norm))
        else:
            # Learnable adjacency
            self.adj_matrix = nn.Parameter(torch.rand(n_roi, n_roi) * 0.1)
            if graph_type == 'cheb':
                # Will compute dynamically
                self.laplacian = None
        
        # DropEdge
        self.drop_edge = DropEdge(p=drop_edge_p)
        
        # Graph convolution layers
        self.graph_layers = nn.ModuleList()
        for i in range(n_layers):
            if graph_type == 'cheb':
                self.graph_layers.append(ChebConv(d_latent, d_latent, K=cheb_k))
            elif graph_type == 'gat':
                self.graph_layers.append(GraphAttentionLayer(d_latent, d_latent, n_heads=n_heads, dropout=dropout))
            else:
                raise ValueError(f"Unknown graph_type: {graph_type}")
        
        # Layer norms
        self.norms = nn.ModuleList([nn.LayerNorm(d_latent) for _ in range(n_layers)])
        
        # Temporal positional encoding (optional)
        if use_temporal_encoding:
            self.temporal_proj = nn.Linear(d_latent, d_latent)
        
        # Output projection: node features → temporal embeddings
        self.output_proj = nn.Sequential(
            nn.Linear(d_latent * n_roi, d_latent),
            nn.GELU(),
            nn.LayerNorm(d_latent)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def get_adjacency(self) -> torch.Tensor:
        """Get adjacency matrix (symmetrize if learnable)"""
        if isinstance(self.adj_matrix, nn.Parameter):
            # Make symmetric
            adj = (self.adj_matrix + self.adj_matrix.T) / 2
            # Apply sigmoid to bound
            adj = torch.sigmoid(adj)
            return adj
        else:
            return self.adj_matrix
    
    def forward(self, x: torch.Tensor, fc_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, N_ROI] ROI time series
            fc_matrix: [B, N_ROI, N_ROI] or [N_ROI, N_ROI] Optional batch-specific FC matrices
                      If provided, overrides the default adjacency matrix
        
        Returns:
            z: [B, T, d_latent] Latent spatial embeddings
        """
        B, T, N = x.shape
        
        # Get adjacency (use provided FC if available)
        if fc_matrix is not None:
            # Use provided FC matrix (can be batch-specific)
            if fc_matrix.dim() == 2:
                # Single FC for whole batch [N, N]
                adj = fc_matrix
            elif fc_matrix.dim() == 3:
                # Batch-specific FC [B, N, N]
                # For now, use batch average (or could process separately)
                adj = fc_matrix.mean(dim=0)
            # Drop negative connections (set to 0), then normalize
            adj = torch.clamp(adj, min=0.0)  # Only keep positive connections
            adj = adj / (adj.max() + 1e-8)  # Normalize to [0, 1]
        else:
            # Use default adjacency
            adj = self.get_adjacency()
        
        adj = self.drop_edge(adj)
        
        # Reshape: treat each (B, T) as batch dimension, N as graph nodes
        # [B, T, N] → [B*T, N, 1] → [B*T, N, d_latent]
        x_flat = x.reshape(B * T, N, 1)
        h = self.input_proj(x_flat)  # [B*T, N, d_latent]
        
        # Apply graph convolution layers
        for i, layer in enumerate(self.graph_layers):
            h_residual = h
            
            if self.graph_type == 'cheb':
                # Get Laplacian
                if self.laplacian is None:
                    laplacian = compute_normalized_laplacian(adj)
                else:
                    laplacian = self.laplacian
                
                h = layer(h, laplacian)
            else:  # GAT
                h = layer(h, adj)
            
            # Residual + norm
            h = self.norms[i](h + h_residual)
            h = self.dropout(h)
        
        # Flatten node features: [B*T, N, d_latent] → [B*T, N*d_latent]
        h_flat = h.reshape(B * T, N * self.d_latent)
        
        # Project to temporal embeddings
        z_flat = self.output_proj(h_flat)  # [B*T, d_latent]
        
        # Reshape back: [B*T, d_latent] → [B, T, d_latent]
        z = z_flat.reshape(B, T, self.d_latent)
        
        # Optional temporal encoding
        if self.use_temporal_encoding:
            # Simple learned temporal projection
            z = z + self.temporal_proj(z)
        
        return z


if __name__ == "__main__":
    # Test real graph encoder
    print("Real Graph-Based Spatial Encoder")
    print("=" * 60)
    
    B, T, N_ROI = 4, 100, 116
    d_latent = 128
    
    # Create random FC matrix
    fc_matrix = torch.randn(N_ROI, N_ROI)
    fc_matrix = (fc_matrix + fc_matrix.T) / 2  # Symmetric
    fc_matrix = torch.abs(fc_matrix)  # Positive
    
    # Test ChebNet
    print("\n1. ChebNet (Spectral Convolution)")
    encoder_cheb = GraphSpatialEncoderReal(
        n_roi=N_ROI,
        d_latent=d_latent,
        n_layers=3,
        graph_type='cheb',
        cheb_k=3,
        drop_edge_p=0.1,
        fc_matrix=fc_matrix
    )
    
    x = torch.randn(B, T, N_ROI)
    z_cheb = encoder_cheb(x)
    
    print(f"  Input: {x.shape}")
    print(f"  Output: {z_cheb.shape}")
    print(f"  Parameters: {sum(p.numel() for p in encoder_cheb.parameters()):,}")
    
    # Test GAT
    print("\n2. GAT (Graph Attention)")
    encoder_gat = GraphSpatialEncoderReal(
        n_roi=N_ROI,
        d_latent=d_latent,
        n_layers=3,
        graph_type='gat',
        n_heads=4,
        drop_edge_p=0.1,
        fc_matrix=fc_matrix
    )
    
    z_gat = encoder_gat(x)
    
    print(f"  Input: {x.shape}")
    print(f"  Output: {z_gat.shape}")
    print(f"  Parameters: {sum(p.numel() for p in encoder_gat.parameters()):,}")
    
    # Test DropEdge
    print("\n3. DropEdge Effect")
    drop_edge = DropEdge(p=0.3)
    
    adj_original = torch.rand(10, 10)
    adj_original = (adj_original + adj_original.T) / 2
    
    drop_edge.eval()
    adj_eval = drop_edge(adj_original)
    print(f"  Eval mode: edges dropped = {(adj_eval == 0).sum().item()}")
    
    drop_edge.train()
    adj_train = drop_edge(adj_original)
    print(f"  Train mode: edges dropped = {(adj_train == 0).sum().item()}")
    
    # Test Laplacian computation
    print("\n4. Graph Laplacian")
    L = compute_normalized_laplacian(fc_matrix[:10, :10])
    eigenvals = torch.linalg.eigvalsh(L)
    print(f"  Laplacian shape: {L.shape}")
    print(f"  Eigenvalue range: [{eigenvals.min():.4f}, {eigenvals.max():.4f}]")
    print(f"  Is symmetric: {torch.allclose(L, L.T)}")
