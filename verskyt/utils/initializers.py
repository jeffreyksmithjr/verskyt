"""
Custom initializers for Tversky neural network parameters.

Based on empirical findings from the paper's experiments.
"""

from typing import Optional

import torch
import torch.nn as nn


def uniform_init(tensor: torch.Tensor, a: float = -1.0, b: float = 1.0):
    """
    Uniform initialization.

    Paper finding: Uniform initialization led to higher convergence
    probability for XOR compared to normal and orthogonal.
    """
    with torch.no_grad():
        tensor.uniform_(a, b)
    return tensor


def xavier_uniform_init(tensor: torch.Tensor, gain: float = 1.0):
    """Xavier/Glorot uniform initialization."""
    nn.init.xavier_uniform_(tensor, gain=gain)
    return tensor


def initialize_for_xor(
    prototypes: torch.Tensor,
    features: torch.Tensor,
    seed: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize parameters specifically for XOR task.

    Based on the paper's Figure 1 showing a working XOR configuration:
    - 2 prototypes: p0 = {}, p1 = {f0, f1}
    - 2 features positioned to separate XOR classes

    Args:
        prototypes: Prototype tensor to initialize [2, 2]
        features: Feature tensor to initialize [num_features, 2]
        seed: Random seed for reproducibility

    Returns:
        Tuple of (initialized_prototypes, initialized_features)
    """
    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        # Initialize based on paper's working configuration
        if features.shape[0] >= 2:
            # Feature vectors from paper Figure 1
            # f0 points toward [1, 0] region
            # f1 points toward [0, 1] region
            features[0] = torch.tensor([1.0, -0.5])
            features[1] = torch.tensor([-0.5, 1.0])

            # Additional features if num_features > 2
            if features.shape[0] > 2:
                features[2:].uniform_(-0.5, 0.5)
        else:
            # Single feature case (paper shows this can work)
            features[0] = torch.tensor([1.0, 1.0])

        # Prototypes from paper
        # p0 represents class 0 (inputs [0,0] and [1,1])
        # p1 represents class 1 (inputs [0,1] and [1,0])
        if prototypes.shape[0] >= 2:
            # p0 has no features strongly (near origin)
            prototypes[0] = torch.tensor([0.1, 0.1])
            # p1 has both features (away from origin)
            prototypes[1] = torch.tensor([0.5, 0.5])

    return prototypes, features


def smart_init(layer: nn.Module, method: str = "xavier_uniform", **kwargs):
    """
    Smart initialization for Tversky layers based on paper findings.

    Args:
        layer: TverskyProjectionLayer or TverskySimilarityLayer
        method: Initialization method
        **kwargs: Additional arguments for initialization
    """
    from verskyt.layers import TverskyProjectionLayer, TverskySimilarityLayer

    if isinstance(layer, (TverskyProjectionLayer, TverskySimilarityLayer)):
        if method == "uniform":
            if hasattr(layer, "prototypes"):
                uniform_init(layer.prototypes, **kwargs)
            uniform_init(layer.feature_bank, **kwargs)
        elif method == "xavier_uniform":
            if hasattr(layer, "prototypes"):
                xavier_uniform_init(layer.prototypes, **kwargs)
            xavier_uniform_init(layer.feature_bank, **kwargs)
        elif method == "xor" and hasattr(layer, "prototypes"):
            # Special initialization for XOR task
            initialize_for_xor(layer.prototypes, layer.feature_bank)
    else:
        # Fall back to standard PyTorch initialization
        if method == "uniform":
            nn.init.uniform_(layer.weight, **kwargs)
        elif method == "xavier_uniform":
            nn.init.xavier_uniform_(layer.weight, **kwargs)
