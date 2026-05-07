"""
BrainMech foundation model components
"""

# V2 Core Components
from .graph_spatial_encoder import GraphSpatialEncoderReal, ChebConv, GraphAttentionLayer
from .temporal_transformer import TemporalTransformer, PositionalEncoding
from .episodic_feedback import EpisodicFeedbackTransformer
from .task_context_encoder import TaskContextEncoder
from .control_fusion import ControlFusion
from .koopman_core_structured import DPLRKoopmanCore
from .decoder_stochastic import StochasticDecoder, CognitiveStateReadoutStochastic
from .regularizers import SpectralRegularizer, GraphPriorRegularizer, ConsistencyRegularizer

# V2 Models
from .brainmech import BrainMech

# V2 Extensions
from .masked_pretraining import MaskedAutoencoderWrapper

__all__ = [
    # Graph Encoders
    'GraphSpatialEncoderReal',
    'ChebConv',
    'GraphAttentionLayer',
    
    # Temporal
    'TemporalTransformer',
    'PositionalEncoding',
    
    # Control System
    'EpisodicFeedbackTransformer',
    'TaskContextEncoder',
    'ControlFusion',
    
    # Koopman Cores
    'DPLRKoopmanCore',
    
    # Decoders
    'StochasticDecoder',
    'CognitiveStateReadoutStochastic',
    
    # Regularizers
    'SpectralRegularizer',
    'GraphPriorRegularizer',
    'ConsistencyRegularizer',
    
    # Main Model
    'BrainMech',
    
    # Pretraining
    'MaskedAutoencoderWrapper',
]

