"""
Tversky neural network layers.

Implements TverskySimilarityLayer (Equation 6) and TverskyProjectionLayer (Equation 7)
from the paper.
"""

from typing import Literal, Optional, Union

import torch
import torch.nn as nn

from verskyt.core.similarity import (
    DifferenceReduction,
    IntersectionReduction,
    tversky_contrast_similarity,
    tversky_similarity,
)


class TverskySimilarityLayer(nn.Module):
    """
    Tversky Similarity Layer (Equation 6 from paper).

    Computes similarity between two objects using learnable feature bank
    and Tversky parameters (α, β, θ).

    S_Ω,α,β,θ(a,b): ℝ^d × ℝ^d → ℝ
    """

    def __init__(
        self,
        in_features: int,
        num_features: int,
        alpha: float = 0.5,
        beta: float = 0.5,
        learnable_ab: bool = True,
        learnable_theta: bool = False,
        theta: float = 1e-7,
        intersection_reduction: Union[IntersectionReduction, str] = "product",
        difference_reduction: Union[DifferenceReduction, str] = "substractmatch",
        use_contrast_form: bool = False,
        feature_init: Literal[
            "uniform", "normal", "xavier_uniform", "xavier_normal"
        ] = "xavier_uniform",
    ):
        """
        Initialize Tversky Similarity Layer.

        Args:
            in_features: Dimension of input vectors
            num_features: Number of features in feature bank (|Ω|)
            alpha: Initial value for α parameter (weight for a's distinctive features)
            beta: Initial value for β parameter (weight for b's distinctive features)
            learnable_ab: Whether α and β are learnable parameters
            learnable_theta: Whether θ is a learnable parameter (only for contrast form)
            theta: Initial value or constant for numerical stability
            intersection_reduction: Method for computing feature intersections
            difference_reduction: Method for computing feature differences
            use_contrast_form: Use linear combination instead of ratio form
            feature_init: Initialization method for feature bank
        """
        super().__init__()

        self.in_features = in_features
        self.num_features = num_features
        self.intersection_reduction = intersection_reduction
        self.difference_reduction = difference_reduction
        self.use_contrast_form = use_contrast_form

        # Initialize feature bank Ω
        self.feature_bank = nn.Parameter(torch.empty(num_features, in_features))

        # Initialize Tversky parameters
        if learnable_ab:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
            self.beta = nn.Parameter(torch.tensor(float(beta)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha)))
            self.register_buffer("beta", torch.tensor(float(beta)))

        # Theta parameter (for numerical stability or contrast form)
        if use_contrast_form and learnable_theta:
            self.theta = nn.Parameter(torch.tensor(float(theta)))
        else:
            self.register_buffer("theta", torch.tensor(float(theta)))

        # Initialize parameters
        self.feature_init = feature_init
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters according to specified method."""
        if self.feature_init == "uniform":
            # Uniform initialization as used in paper's XOR experiments
            nn.init.uniform_(self.feature_bank, -1, 1)
        elif self.feature_init == "normal":
            nn.init.normal_(self.feature_bank, std=0.02)
        elif self.feature_init == "xavier_uniform":
            nn.init.xavier_uniform_(self.feature_bank)
        elif self.feature_init == "xavier_normal":
            nn.init.xavier_normal_(self.feature_bank)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute element-wise Tversky similarity between objects a and b.

        Args:
            a: First object tensor of shape [batch_size, in_features]
            b: Second object tensor of shape [batch_size, in_features]

        Returns:
            Similarity scores of shape [batch_size]
        """
        batch_size = a.shape[0]

        # Compute similarities element-wise
        similarities = []
        for i in range(batch_size):
            a_i = a[i : i + 1]  # Keep batch dimension
            b_i = b[i : i + 1]  # Keep batch dimension

            if self.use_contrast_form:
                # Use linear combination form
                sim = tversky_contrast_similarity(
                    a_i,
                    b_i,
                    self.feature_bank,
                    self.alpha,
                    self.beta,
                    self.theta,
                    self.intersection_reduction,
                    self.difference_reduction,
                )
            else:
                # Use ratio (Tversky Index) form
                sim = tversky_similarity(
                    a_i,
                    b_i,
                    self.feature_bank,
                    self.alpha,
                    self.beta,
                    self.theta.item(),
                    self.intersection_reduction,
                    self.difference_reduction,
                )

            similarities.append(sim[0, 0])  # Extract scalar similarity

        return torch.stack(similarities)


class TverskyProjectionLayer(nn.Module):
    """A projection layer based on Tversky similarity (Equation 7 from paper).

    This layer replaces standard linear projections by computing Tversky similarity
    between inputs and learned prototype vectors. Unlike linear layers, it can
    model non-linear functions like XOR with a single layer, making it suitable
    for complex pattern recognition tasks.

    The layer implements: P_Ω,α,β,θ,Π(a): ℝ^d → ℝ^p

    Where:
    - Ω: Learnable feature bank of shape [num_features, in_features]
    - Π: Learnable prototype vectors of shape [num_prototypes, in_features]
    - α, β: Asymmetry parameters controlling feature distinctiveness weights
    - θ: Numerical stability constant

    This layer can serve as a drop-in replacement for nn.Linear in many architectures,
    offering improved interpretability and non-linear modeling capabilities.

    Attributes:
        prototypes (nn.Parameter): Learnable prototype vectors of shape
            [num_prototypes, in_features].
        feature_bank (nn.Parameter): Learnable feature bank of shape
            [num_features, in_features].
        alpha (nn.Parameter or torch.Tensor): Tversky weight for input-distinctive features.
        beta (nn.Parameter or torch.Tensor): Tversky weight for prototype-distinctive features.
        bias (nn.Parameter or None): Optional bias term of shape [num_prototypes].
    """

    def __init__(
        self,
        in_features: int,
        num_prototypes: int,
        num_features: int,
        alpha: float = 0.5,
        beta: float = 0.5,
        learnable_ab: bool = True,
        theta: float = 1e-7,
        intersection_reduction: Union[IntersectionReduction, str] = "product",
        difference_reduction: Union[DifferenceReduction, str] = "substractmatch",
        normalize_features: bool = False,
        normalize_prototypes: bool = False,
        prototype_init: Literal[
            "uniform", "normal", "xavier_uniform", "xavier_normal"
        ] = "xavier_uniform",
        feature_init: Literal[
            "uniform", "normal", "xavier_uniform", "xavier_normal"
        ] = "xavier_uniform",
        shared_feature_bank: Optional[nn.Parameter] = None,
        bias: bool = False,
    ):
        """Initialize Tversky Projection Layer.

        Args:
            in_features (int): Size of each input sample's embedding dimension.
            num_prototypes (int): Number of prototype vectors to learn. This typically
                corresponds to the output dimension or number of classes.
            num_features (int): Size of the shared feature bank (|Ω|). This is a key
                hyperparameter controlling the expressiveness of the feature space.
            alpha (float, optional): Initial Tversky weight for input-distinctive
                features (x \\ π). Higher values increase sensitivity to features
                present in input but not in prototypes. Defaults to 0.5.
            beta (float, optional): Initial Tversky weight for prototype-distinctive
                features (π \\ x). Higher values increase sensitivity to features
                present in prototypes but not in input. Defaults to 0.5.
            learnable_ab (bool, optional): Whether α and β are learnable parameters.
                If False, they remain fixed at initial values. Defaults to True.
            theta (float, optional): Small constant for numerical stability in
                similarity computation. Defaults to 1e-7.
            intersection_reduction (Union[IntersectionReduction, str], optional):
                Method for aggregating feature intersections. Options: "product",
                "min", "max", "mean", "gmean", "softmin". Defaults to "product".
            difference_reduction (Union[DifferenceReduction, str], optional):
                Method for computing feature differences. Options: "ignorematch",
                "substractmatch". Defaults to "substractmatch".
            normalize_features (bool, optional): Whether to L2-normalize feature
                bank vectors during forward pass. Defaults to False.
            normalize_prototypes (bool, optional): Whether to L2-normalize input
                and prototype vectors during forward pass. Defaults to False.
            prototype_init (Literal, optional): Initialization method for prototype
                vectors. Options: "uniform", "normal", "xavier_uniform",
                "xavier_normal". Defaults to "xavier_uniform".
            feature_init (Literal, optional): Initialization method for feature bank.
                Same options as prototype_init. Defaults to "xavier_uniform".
            shared_feature_bank (Optional[nn.Parameter], optional): Pre-existing
                feature bank to share across layers. If provided, feature_init
                is ignored. Defaults to None.
            bias (bool, optional): Whether to include a learnable bias term of
                shape [num_prototypes]. Defaults to False.

        Example:
            >>> # Create a projection layer as drop-in replacement for nn.Linear
            >>> layer = TverskyProjectionLayer(
            ...     in_features=128,
            ...     num_prototypes=10,  # like nn.Linear(128, 10)
            ...     num_features=64,    # internal feature space size
            ...     learnable_ab=True
            ... )
            >>> x = torch.randn(32, 128)  # batch of 32 samples
            >>> output = layer(x)         # shape: [32, 10]
        """
        super().__init__()

        self.in_features = in_features
        self.num_prototypes = num_prototypes
        self.num_features = num_features
        self.theta = theta
        self.intersection_reduction = intersection_reduction
        self.difference_reduction = difference_reduction
        self.normalize_features = normalize_features
        self.normalize_prototypes = normalize_prototypes

        # Initialize prototypes Π
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, in_features))

        # Initialize or share feature bank Ω
        if shared_feature_bank is not None:
            # Share feature bank with another layer
            self.feature_bank = shared_feature_bank
            self.shared_features = True
        else:
            self.feature_bank = nn.Parameter(torch.empty(num_features, in_features))
            self.shared_features = False

        # Initialize Tversky parameters
        if learnable_ab:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
            self.beta = nn.Parameter(torch.tensor(float(beta)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha)))
            self.register_buffer("beta", torch.tensor(float(beta)))

        # Optional bias term
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_prototypes))
        else:
            self.register_buffer("bias", None)

        # Store initialization methods
        self.prototype_init = prototype_init
        self.feature_init = feature_init

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters according to specified methods."""
        # Initialize prototypes
        if self.prototype_init == "uniform":
            nn.init.uniform_(self.prototypes, -1, 1)
        elif self.prototype_init == "normal":
            nn.init.normal_(self.prototypes, std=0.02)
        elif self.prototype_init == "xavier_uniform":
            nn.init.xavier_uniform_(self.prototypes)
        elif self.prototype_init == "xavier_normal":
            nn.init.xavier_normal_(self.prototypes)

        # Initialize feature bank (only if not shared)
        if not self.shared_features:
            if self.feature_init == "uniform":
                nn.init.uniform_(self.feature_bank, -1, 1)
            elif self.feature_init == "normal":
                nn.init.normal_(self.feature_bank, std=0.02)
            elif self.feature_init == "xavier_uniform":
                nn.init.xavier_uniform_(self.feature_bank)
            elif self.feature_init == "xavier_normal":
                nn.init.xavier_normal_(self.feature_bank)

        # Initialize bias if present
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass through the Tversky projection layer.

        Projects the input to prototype similarity space by computing Tversky
        similarity between each input and all learned prototype vectors.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_features].

        Returns:
            torch.Tensor: Tversky similarity scores of shape [batch_size, num_prototypes].
                Values are in [0, 1] range for standard Tversky Index formulation,
                representing similarity to each prototype.

        Note:
            This layer can serve as a drop-in replacement for nn.Linear, but
            produces similarity-based rather than linear projections.
        """
        # Compute Tversky similarity to all prototypes
        similarity = tversky_similarity(
            x,
            self.prototypes,
            self.feature_bank,
            self.alpha,
            self.beta,
            self.theta,
            self.intersection_reduction,
            self.difference_reduction,
            self.normalize_features,
            self.normalize_prototypes,
        )

        # Add bias if present
        if self.bias is not None:
            similarity = similarity + self.bias

        return similarity

    def get_prototype(self, index: int) -> torch.Tensor:
        """Get a specific prototype vector."""
        return self.prototypes[index].detach().clone()

    def set_prototype(self, index: int, value: torch.Tensor):
        """Set a specific prototype vector."""
        with torch.no_grad():
            self.prototypes[index] = value

    def get_feature(self, index: int) -> torch.Tensor:
        """Get a specific feature vector."""
        return self.feature_bank[index].detach().clone()

    def set_feature(self, index: int, value: torch.Tensor):
        """Set a specific feature vector."""
        with torch.no_grad():
            self.feature_bank[index] = value

    @property
    def weight(self):
        """Compatibility property for drop-in replacement of nn.Linear."""
        # Return prototypes as 'weight' for compatibility
        # Note: This is not equivalent to linear layer weights
        return self.prototypes

    def extra_repr(self) -> str:
        """String representation with layer configuration."""
        s = (
            f"in_features={self.in_features}, "
            f"num_prototypes={self.num_prototypes}, "
            f"num_features={self.num_features}"
        )
        if self.bias is not None:
            s += ", bias=True"
        if self.shared_features:
            s += ", shared_features=True"
        return s
