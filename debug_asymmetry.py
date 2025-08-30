"""
Debug asymmetry test in detail.
"""

import torch

from verskyt.core.similarity import tversky_similarity


def debug_asymmetry_detailed():
    """Debug the asymmetry case step by step."""
    print("=== Debugging Asymmetry with ignorematch ===")
    x = torch.tensor([[1.0, 0.5]])
    prototypes = torch.tensor([[0.5, 1.0]])
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    print(f"x: {x}")
    print(f"prototypes: {prototypes}")
    print(f"features: {features}")

    # Feature memberships
    x_membership = torch.einsum("bi,fi->bf", x, features)
    p_membership = torch.einsum("pi,fi->pf", prototypes, features)

    print(f"x_membership: {x_membership}")  # [[1.0, 0.5]]
    print(f"p_membership: {p_membership}")  # [[0.5, 1.0]]

    # Forward similarity with α=0.8, β=0.2
    sim_forward = tversky_similarity(
        x,
        prototypes,
        features,
        alpha=0.8,
        beta=0.2,
        theta=1e-7,
        difference_reduction="ignorematch",
    )

    # Reverse similarity with α=0.2, β=0.8
    sim_reverse = tversky_similarity(
        prototypes,
        x,
        features,
        alpha=0.2,
        beta=0.8,
        theta=1e-7,
        difference_reduction="ignorematch",
    )

    print(f"Forward similarity (α=0.8, β=0.2): {sim_forward}")
    print(f"Reverse similarity (α=0.2, β=0.8): {sim_reverse}")
    print(f"Different? {not torch.allclose(sim_forward, sim_reverse, atol=1e-3)}")

    # Let's manually compute to understand:
    # With ignorematch:
    # - intersection: min(1, 0.5) * min(0.5, 1) = 0.5 * 0.5 = 0.25 (if using min reduction)
    # - x_minus_p: features where x > 0 and p <= 0. Here: none (both have positive memberships)
    # - p_minus_x: features where p > 0 and x <= 0. Here: none

    # So both should be: 0.25 / (0.25 + 0) = 1.0

    print("\n=== Test with clear asymmetric case ===")
    # Make a more asymmetric case
    x_asym = torch.tensor([[1.0, 0.0]])
    p_asym = torch.tensor([[0.0, 0.5]])

    x_mem = torch.einsum("bi,fi->bf", x_asym, features)
    p_mem = torch.einsum("pi,fi->pf", p_asym, features)
    print(f"x_membership: {x_mem}")  # [[1.0, 0.0]]
    print(f"p_membership: {p_mem}")  # [[0.0, 0.5]]

    # Forward: x has feature 0, p doesn't; p has feature 1, x doesn't
    sim_f = tversky_similarity(
        x_asym,
        p_asym,
        features,
        alpha=0.8,
        beta=0.2,
        theta=1e-7,
        difference_reduction="ignorematch",
    )

    # Reverse: p has feature 1, x doesn't; x has feature 0, p doesn't
    sim_r = tversky_similarity(
        p_asym,
        x_asym,
        features,
        alpha=0.2,
        beta=0.8,
        theta=1e-7,
        difference_reduction="ignorematch",
    )

    print(f"Asymmetric forward: {sim_f}")
    print(f"Asymmetric reverse: {sim_r}")
    print(f"Different? {not torch.allclose(sim_f, sim_r, atol=1e-3)}")


if __name__ == "__main__":
    debug_asymmetry_detailed()
