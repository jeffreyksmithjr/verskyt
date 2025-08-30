"""
Test the new asymmetry test.
"""

import torch

from verskyt.core.similarity import tversky_similarity


def test_new_asymmetry():
    x = torch.tensor([[1.0, 0.0]])  # Has first feature, not second
    prototypes = torch.tensor([[0.0, 1.0]])  # Has second feature, not first
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    # Test asymmetry by varying α,β weights for same comparison
    sim_alpha_high = tversky_similarity(
        x,
        prototypes,
        features,
        alpha=0.9,
        beta=0.1,
        theta=1e-7,
        difference_reduction="ignorematch",
    )

    sim_beta_high = tversky_similarity(
        x,
        prototypes,
        features,
        alpha=0.1,
        beta=0.9,
        theta=1e-7,
        difference_reduction="ignorematch",
    )

    print(f"Alpha high (0.9, 0.1): {sim_alpha_high}")
    print(f"Beta high (0.1, 0.9): {sim_beta_high}")
    print(f"Different? {not torch.allclose(sim_alpha_high, sim_beta_high, atol=1e-3)}")

    # Let's manually compute:
    # intersection = 0 (no common features)
    # x_minus_p = 1 (x has feature 0, p doesn't)
    # p_minus_x = 1 (p has feature 1, x doesn't)

    # sim_alpha_high: 0 / (0 + 0.9*1 + 0.1*1) = 0 / 1.0 = 0
    # sim_beta_high: 0 / (0 + 0.1*1 + 0.9*1) = 0 / 1.0 = 0
    # These are the same!

    print("\nExpected: both should be 0 since no intersection")
    print("The asymmetry shows up when there IS some intersection...")


if __name__ == "__main__":
    test_new_asymmetry()
