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
    """Methods for reducing feature intersections (A ∩ B).

    These methods determine how to aggregate feature membership scores when
    computing the intersection between two objects in the feature space.

    Attributes:
        PRODUCT: Element-wise product of membership scores (default in paper).
        MIN: Element-wise minimum of membership scores.
        MAX: Element-wise maximum of membership scores.
        MEAN: Element-wise arithmetic mean of membership scores.
        GMEAN: Element-wise geometric mean of membership scores.
        SOFTMIN: Differentiable soft minimum using LogSumExp.
    """

    PRODUCT = "product"
    MIN = "min"
    MAX = "max"
    MEAN = "mean"
    GMEAN = "gmean"
    SOFTMIN = "softmin"


class DifferenceReduction(str, Enum):
    """Methods for reducing feature differences (A - B).

    These methods determine how to compute the distinctive features of one
    object compared to another, implementing Equations 4 and 5 from the paper.

    Attributes:
        IGNOREMATCH: Only count features present in A but not in B (Equation 4).
        SUBSTRACTMATCH: Account for magnitude differences in shared features
            (Equation 5).
    """

    IGNOREMATCH = "ignorematch"  # Only features in A but not in B
    SUBSTRACTMATCH = "substractmatch"  # Account for magnitude differences


def compute_feature_membership(x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """Compute feature membership scores for objects.

    This function computes how much each feature from the feature bank is
    present in each object. The membership score is computed as the dot
    product between object vectors and feature vectors.

    Args:
        x (torch.Tensor): Object vectors of shape [batch_size, in_features] or
            [num_objects, in_features].
        features (torch.Tensor): Feature bank of shape [num_features, in_features].

    Returns:
        torch.Tensor: Membership scores of shape [batch_size, num_features] or
            [num_objects, num_features]. Higher values indicate stronger presence
            of the feature in the object.

    Note:
        Implementation uses Einstein summation for efficient computation:
        x·fₖ represents the measure of feature fₖ in object x (from paper).
    """
    # Equation from paper: x·fₖ represents measure of feature fₖ in x
    return torch.einsum("bi,fi->bf", x, features)


def compute_salience(x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """Compute salience of objects (Equation 2 from paper).

    Salience measures the total amount of features present in an object.
    It is computed as the sum of all positive feature membership scores,
    representing the psychological notion of an object's perceptual prominence.

    Args:
        x (torch.Tensor): Object vectors of shape [batch_size, in_features].
        features (torch.Tensor): Feature bank of shape [num_features, in_features].

    Returns:
        torch.Tensor: Salience scores of shape [batch_size]. Higher values
            indicate objects with more prominent features.

    Note:
        Only positive memberships are summed, as negative values indicate
        absence of features. Implements Equation 2: salience(x) = Σᵢ max(0, x·fᵢ).
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
        # Broadcast to ensure same shape before stacking
        broadcasted_neg_x = neg_x.expand_as(x_pos * p_pos)
        broadcasted_neg_p = neg_p.expand_as(x_pos * p_pos)
        stacked = torch.stack([broadcasted_neg_x, broadcasted_neg_p], dim=-1)
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
    elif method == DifferenceReduction.SUBSTRACTMATCH or method == "substractmatch":
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
    """Compute Tversky similarity between inputs and prototypes.

    This is the core function implementing differentiable Tversky similarity
    for neural networks. It computes psychologically-motivated similarity
    scores that account for both common and distinctive features.

    The function implements the Tversky Index formulation (Equation 1):
    TI(a,b) = f(A∩B) / (f(A∩B) + αf(A-B) + βf(B-A))

    Where:
    - f(A∩B): Intersection - features common to both objects
    - f(A-B): Features distinctive to object A
    - f(B-A): Features distinctive to object B
    - α, β: Asymmetry parameters (α=β=0.5 gives Jaccard similarity)

    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, in_features].
        prototypes (torch.Tensor): Prototype tensor of shape
            [num_prototypes, in_features].
        feature_bank (torch.Tensor): Feature bank tensor of shape
            [num_features, in_features].
        alpha (Union[torch.Tensor, float]): Weight for x's distinctive features.
            Higher values make the similarity more sensitive to features in x
            but not in prototypes. Must be ≥ 0.
        beta (Union[torch.Tensor, float]): Weight for prototype's distinctive
            features. Higher values make the similarity more sensitive to features
            in prototypes but not in x. Must be ≥ 0.
        theta (float, optional): Small constant for numerical stability.
            Defaults to 1e-7.
        intersection_reduction (Union[IntersectionReduction, str], optional):
            Method for computing feature intersections. Defaults to "product".
        difference_reduction (Union[DifferenceReduction, str], optional):
            Method for computing feature differences. Defaults to "substractmatch".
        normalize_features (bool, optional): Whether to L2-normalize feature vectors.
            Defaults to False.
        normalize_prototypes (bool, optional): Whether to L2-normalize input and
            prototype vectors. Defaults to False.

    Returns:
        torch.Tensor: Similarity scores of shape [batch_size, num_prototypes].
            Values are in [0, 1] range, with 1 indicating perfect similarity.

    Raises:
        ValueError: If alpha or beta are negative, or if reduction methods are invalid.

    Example:
        >>> import torch
        >>> x = torch.randn(2, 4)  # 2 samples, 4 dimensions
        >>> prototypes = torch.randn(3, 4)  # 3 prototypes
        >>> features = torch.randn(8, 4)  # 8 features
        >>> similarity = tversky_similarity(x, prototypes, features, 0.5, 0.5)
        >>> similarity.shape
        torch.Size([2, 3])
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
    intersection = _compute_intersection(x_mem_exp, p_mem_exp, intersection_reduction)

    # Step 4: Calculate differences f(A - B) and f(B - A)
    x_minus_p = _compute_difference(x_mem_exp, p_mem_exp, difference_reduction, True)
    p_minus_x = _compute_difference(x_mem_exp, p_mem_exp, difference_reduction, False)

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
    """Compute Tversky contrast model similarity (alternative formulation).

    This function implements the linear combination form of Tversky similarity
    rather than the normalized ratio form. It may be preferred when working
    with raw similarity scores or when the denominator normalization is
    handled elsewhere in the model.

    Uses the linear combination from Equation 1:
    S(a,b) = θf(A∩B) - αf(A-B) - βf(B-A)

    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, in_features].
        prototypes (torch.Tensor): Prototype tensor of shape
            [num_prototypes, in_features].
        feature_bank (torch.Tensor): Feature bank tensor of shape
            [num_features, in_features].
        alpha (Union[torch.Tensor, float]): Weight for x's distinctive features.
        beta (Union[torch.Tensor, float]): Weight for prototype's distinctive features.
        theta (Union[torch.Tensor, float], optional): Weight for common features.
            Defaults to 1.0.
        intersection_reduction (Union[IntersectionReduction, str], optional):
            Method for computing intersections. Defaults to "product".
        difference_reduction (Union[DifferenceReduction, str], optional):
            Method for computing differences. Defaults to "substractmatch".

    Returns:
        torch.Tensor: Similarity scores of shape [batch_size, num_prototypes].
            Unlike the Tversky Index, these scores are not bounded to [0,1]
            and can be negative.

    Note:
        This formulation is useful when you want the raw contrast scores
        without normalization, or when combining with other similarity measures.
    """
    # Compute feature memberships
    x_membership = torch.einsum("bi,fi->bf", x, feature_bank)
    p_membership = torch.einsum("pi,fi->pf", prototypes, feature_bank)

    # Expand dimensions
    x_mem_exp = x_membership.unsqueeze(1)
    p_mem_exp = p_membership.unsqueeze(0)

    # Calculate components
    intersection = _compute_intersection(x_mem_exp, p_mem_exp, intersection_reduction)
    x_minus_p = _compute_difference(x_mem_exp, p_mem_exp, difference_reduction, True)
    p_minus_x = _compute_difference(x_mem_exp, p_mem_exp, difference_reduction, False)

    # Linear combination form
    similarity = theta * intersection - alpha * x_minus_p - beta * p_minus_x

    return similarity
