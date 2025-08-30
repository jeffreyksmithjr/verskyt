"""
Core implementation of differentiable Tversky similarity.

Based on "Tversky Neural Networks: Psychologically Plausible Deep Learning
with Differentiable Tversky Similarity" (Doumbouya et al., 2025).
"""

from enum import Enum
from typing import Union

import torch
import torch.nn.functional as F


class IntersectionReduction(str, Enum):
    """Methods for reducing feature intersections (A ∩ B)."""

    PRODUCT = "product"
    MIN = "min"
    MAX = "max"
    MEAN = "mean"
    GMEAN = "gmean"
    SOFTMIN = "softmin"


class DifferenceReduction(str, Enum):
    """Methods for reducing feature differences (A - B)."""

    IGNOREMATCH = "ignorematch"  # Only features in A but not in B
    SUBSTRACTMATCH = "substractmatch"  # Account for magnitude differences


def compute_feature_membership(
    x: torch.Tensor, features: torch.Tensor
) -> torch.Tensor:
    """
    Compute feature membership scores for objects.

    Args:
        x: Object vectors of shape [batch_size, in_features] or
            [num_objects, in_features]
        features: Feature bank of shape [num_features, in_features]

    Returns:
        Membership scores of shape [batch_size, num_features] or
            [num_objects, num_features]
    """
    # Equation from paper: x·fₖ represents measure of feature fₖ in x
    return torch.einsum("bi,fi->bf", x, features)


def compute_salience(x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """
    Compute salience of objects (Equation 2 from paper).

    Salience is the sum of measures of all features present in the object.

    Args:
        x: Object vectors of shape [batch_size, in_features]
        features: Feature bank of shape [num_features, in_features]

    Returns:
        Salience scores of shape [batch_size]
    """
    membership = compute_feature_membership(x, features)
    # Only sum positive memberships (features present in object)
    positive_membership = F.relu(membership)
    return positive_membership.sum(dim=-1)


def _compute_intersection(
    x_membership: torch.Tensor,
    p_membership: torch.Tensor,
    method: Union[IntersectionReduction, str],
) -> torch.Tensor:
    """
    Compute feature intersection f(A ∩ B) using specified reduction method.

    Args:
        x_membership: Input membership of shape [batch_size, 1, num_features]
        p_membership: Prototype membership of shape [1, num_prototypes, num_features]
        method: Reduction method for intersection

    Returns:
        Intersection scores of shape [batch_size, num_prototypes]
    """
    # Only consider positive memberships (features present)
    x_pos = F.relu(x_membership)
    p_pos = F.relu(p_membership)

    if method == IntersectionReduction.PRODUCT or method == "product":
        # Product of memberships for common features
        intersection_scores = x_pos * p_pos
    elif method == IntersectionReduction.MIN or method == "min":
        # Minimum of memberships
        intersection_scores = torch.minimum(x_pos, p_pos)
    elif method == IntersectionReduction.MAX or method == "max":
        # Maximum of memberships
        intersection_scores = torch.maximum(x_pos, p_pos)
    elif method == IntersectionReduction.MEAN or method == "mean":
        # Mean of memberships
        intersection_scores = (x_pos + p_pos) / 2
    elif method == IntersectionReduction.GMEAN or method == "gmean":
        # Geometric mean
        # Add small epsilon to avoid sqrt(0)
        intersection_scores = torch.sqrt(x_pos * p_pos + 1e-8)
    elif method == IntersectionReduction.SOFTMIN or method == "softmin":
        # Soft minimum using LogSumExp trick
        # softmin(a,b) = -log(exp(-a) + exp(-b))
        neg_x = -x_pos
        neg_p = -p_pos
        stacked = torch.stack([neg_x, neg_p], dim=-1)
        intersection_scores = -torch.logsumexp(stacked, dim=-1)
    else:
        raise ValueError(f"Unknown intersection reduction method: {method}")

    # Sum across features to get total intersection
    return intersection_scores.sum(dim=-1)


def _compute_difference(
    x_membership: torch.Tensor,
    p_membership: torch.Tensor,
    method: Union[DifferenceReduction, str],
    compute_x_minus_p: bool = True,
) -> torch.Tensor:
    """
    Compute feature difference f(A - B) using specified reduction method.

    Args:
        x_membership: Input membership of shape [batch_size, 1, num_features]
        p_membership: Prototype membership of shape [1, num_prototypes, num_features]
        method: Reduction method for difference
        compute_x_minus_p: If True compute (x - p), else compute (p - x)

    Returns:
        Difference scores of shape [batch_size, num_prototypes]
    """
    if compute_x_minus_p:
        a_mem, b_mem = x_membership, p_membership
    else:
        a_mem, b_mem = p_membership, x_membership

    if method == DifferenceReduction.IGNOREMATCH or method == "ignorematch":
        # Features in A but not in B (Equation 4)
        # Only count features where a > 0 and b <= 0
        a_pos = a_mem > 0
        b_neg = b_mem <= 0
        mask = a_pos & b_neg
        difference_scores = F.relu(a_mem) * mask.float()
    elif (
        method == DifferenceReduction.SUBSTRACTMATCH
        or method == "substractmatch"
    ):
        # Account for magnitude differences (Equation 5)
        # Features present in both but greater in A
        diff = a_mem - b_mem
        both_positive = (a_mem > 0) & (b_mem > 0)
        # Only keep positive differences where both have the feature
        difference_scores = F.relu(diff) * both_positive.float()
    else:
        raise ValueError(f"Unknown difference reduction method: {method}")

    # Sum across features
    return difference_scores.sum(dim=-1)


def tversky_similarity(
    x: torch.Tensor,
    prototypes: torch.Tensor,
    feature_bank: torch.Tensor,
    alpha: Union[torch.Tensor, float],
    beta: Union[torch.Tensor, float],
    theta: float = 1e-7,
    intersection_reduction: Union[IntersectionReduction, str] = "product",
    difference_reduction: Union[DifferenceReduction, str] = "substractmatch",
    normalize_features: bool = False,
    normalize_prototypes: bool = False,
) -> torch.Tensor:
    """
    Compute Tversky similarity between inputs and prototypes.

    Implements the differentiable Tversky similarity function from Equation 1:
    S(a,b) = θf(A∩B) - αf(A-B) - βf(B-A)

    The Tversky Index formulation (used in the paper) is:
    TI(a,b) = f(A∩B) / (f(A∩B) + αf(A-B) + βf(B-A))

    Args:
        x: Input tensor of shape [batch_size, in_features]
        prototypes: Prototype tensor of shape [num_prototypes, in_features]
        feature_bank: Feature bank tensor of shape [num_features, in_features]
        alpha: Weight for x's distinctive features (x - prototype)
        beta: Weight for prototype's distinctive features (prototype - x)
        theta: Small constant for numerical stability
        intersection_reduction: Method for reducing intersection measures
        difference_reduction: Method for reducing difference measures
        normalize_features: Whether to normalize feature vectors
        normalize_prototypes: Whether to normalize prototype vectors

    Returns:
        Similarity scores of shape [batch_size, num_prototypes]
    """
    # Optionally normalize vectors (shown to help in some cases per paper)
    if normalize_features:
        feature_bank = F.normalize(feature_bank, p=2, dim=-1)
    if normalize_prototypes:
        prototypes = F.normalize(prototypes, p=2, dim=-1)
        x = F.normalize(x, p=2, dim=-1)

    # Step 1: Compute feature memberships using efficient einsum
    # x_membership: [batch_size, num_features]
    # p_membership: [num_prototypes, num_features]
    x_membership = torch.einsum("bi,fi->bf", x, feature_bank)
    p_membership = torch.einsum("pi,fi->pf", prototypes, feature_bank)

    # Step 2: Expand dimensions for broadcasting
    # x_membership: [batch_size, 1, num_features]
    # p_membership: [1, num_prototypes, num_features]
    x_mem_exp = x_membership.unsqueeze(1)
    p_mem_exp = p_membership.unsqueeze(0)

    # Step 3: Calculate intersection f(A ∩ B)
    intersection = _compute_intersection(
        x_mem_exp, p_mem_exp, intersection_reduction
    )

    # Step 4: Calculate differences f(A - B) and f(B - A)
    x_minus_p = _compute_difference(
        x_mem_exp, p_mem_exp, difference_reduction, True
    )
    p_minus_x = _compute_difference(
        x_mem_exp, p_mem_exp, difference_reduction, False
    )

    # Step 5: Compute final Tversky Index (normalized form used in paper)
    # Ensure alpha and beta are non-negative as per paper
    if isinstance(alpha, torch.Tensor):
        alpha = torch.clamp(alpha, min=0)
    else:
        alpha = max(0, alpha)

    if isinstance(beta, torch.Tensor):
        beta = torch.clamp(beta, min=0)
    else:
        beta = max(0, beta)

    # Tversky Index formulation (Equation 1 normalized form)
    numerator = intersection + theta
    denominator = intersection + alpha * x_minus_p + beta * p_minus_x + theta

    return numerator / denominator


def tversky_contrast_similarity(
    x: torch.Tensor,
    prototypes: torch.Tensor,
    feature_bank: torch.Tensor,
    alpha: Union[torch.Tensor, float],
    beta: Union[torch.Tensor, float],
    theta: Union[torch.Tensor, float] = 1.0,
    intersection_reduction: Union[IntersectionReduction, str] = "product",
    difference_reduction: Union[DifferenceReduction, str] = "substractmatch",
) -> torch.Tensor:
    """
    Compute Tversky contrast model similarity (alternative formulation).

    Uses the linear combination form from Equation 1:
    S(a,b) = θf(A∩B) - αf(A-B) - βf(B-A)

    This is an alternative to the ratio (Tversky Index) form and may be
    useful for certain applications.

    Args:
        x: Input tensor of shape [batch_size, in_features]
        prototypes: Prototype tensor of shape [num_prototypes, in_features]
        feature_bank: Feature bank tensor of shape [num_features, in_features]
        alpha: Weight for x's distinctive features
        beta: Weight for prototype's distinctive features
        theta: Weight for common features
        intersection_reduction: Method for reducing intersection measures
        difference_reduction: Method for reducing difference measures

    Returns:
        Similarity scores of shape [batch_size, num_prototypes]
    """
    # Compute feature memberships
    x_membership = torch.einsum("bi,fi->bf", x, feature_bank)
    p_membership = torch.einsum("pi,fi->pf", prototypes, feature_bank)

    # Expand dimensions
    x_mem_exp = x_membership.unsqueeze(1)
    p_mem_exp = p_membership.unsqueeze(0)

    # Calculate components
    intersection = _compute_intersection(
        x_mem_exp, p_mem_exp, intersection_reduction
    )
    x_minus_p = _compute_difference(
        x_mem_exp, p_mem_exp, difference_reduction, True
    )
    p_minus_x = _compute_difference(
        x_mem_exp, p_mem_exp, difference_reduction, False
    )

    # Linear combination form
    similarity = theta * intersection - alpha * x_minus_p - beta * p_minus_x

    return similarity
