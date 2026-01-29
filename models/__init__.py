"""
BrainMech Brain Model - Architecture
Based on Real Graph Structure, Structured Koopman Operators, and Stochastic Dynamics
"""

# BrainMech Brain Model Core Components
from .graph_spatial_encoder import GraphSpatialEncoder, ChebConv, GraphAttentionLayer
from .temporal_encoder import TemporalTransformer, PositionalEncoding
from .episodic_feedback import EpisodicFeedbackTransformer
from .task_context import TaskContextEncoder
from .control_fusion import ControlFusion
from .koopman_DPLR_core import DPLRKoopmanCore
from .stochastic_decoder import StochasticDecoder
from .regularizers import SpectralRegularizer, GraphPriorRegularizer, ConsistencyRegularizer

# BrainMech Brain Model Models
from .brain_model import BrainMechBrainModel

# BrainMech Brain Model Extensions
from .masked_pretraining import MaskedAutoencoderWrapper

__all__ = [
    # Graph Spatial Encoder
    'GraphSpatialEncoder',
    'ChebConv',
    'GraphAttentionLayer',
    
    # Temporal Encoder
    'TemporalTransformer',
    'PositionalEncoding',
    
    # Control Fusion Module
    'EpisodicFeedbackTransformer',
    'TaskContextEncoder',
    'ControlFusion',
    
    # Structured Koopman Operator (DPLR Koopman Core)
    'DPLRKoopmanCore',
    
    # Stochastic Decoder
    'StochasticDecoder',
    
    # Regularizers (Spectral Regularizer, Graph Prior Regularizer, Consistency Regularizer)
    'SpectralRegularizer',
    'GraphPriorRegularizer',
    'ConsistencyRegularizer',
    
    # Main Models (BrainMech Brain Model)
    'BrainMechBrainModel',
]

