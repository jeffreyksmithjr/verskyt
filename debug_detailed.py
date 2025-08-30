import torch

from verskyt.core.similarity import (
    _compute_difference,
    _compute_intersection,
    tversky_similarity,
)

# Exact test case
x = torch.tensor([[1.0, 0.3]])
prototypes = torch.tensor([[0.3, 1.0]])
features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

# Compute memberships
x_membership = torch.einsum("bi,fi->bf", x, features)
p_membership = torch.einsum("pi,fi->pf", prototypes, features)

print(f"x: {x}")
print(f"prototypes: {prototypes}")
print(f"x_membership: {x_membership}")  # [[1.0, 0.3]]
print(f"p_membership: {p_membership}")  # [[0.3, 1.0]]

# Expand dimensions
x_mem_exp = x_membership.unsqueeze(1)
p_mem_exp = p_membership.unsqueeze(0)

# Compute components
intersection = _compute_intersection(x_mem_exp, p_mem_exp, "product")
x_minus_p = _compute_difference(x_mem_exp, p_mem_exp, "substractmatch", True)
p_minus_x = _compute_difference(x_mem_exp, p_mem_exp, "substractmatch", False)

print(f"\nIntersection (product): {intersection}")
print(f"x - p (substractmatch): {x_minus_p}")
print(f"p - x (substractmatch): {p_minus_x}")

# With substractmatch:
# - Both have positive membership in both features
# - x_minus_p: ReLU(x_mem - p_mem) where both positive
#   Feature 0: ReLU(1.0 - 0.3) = 0.7
#   Feature 1: ReLU(0.3 - 1.0) = 0.0
#   Sum: 0.7
# - p_minus_x: ReLU(p_mem - x_mem) where both positive
#   Feature 0: ReLU(0.3 - 1.0) = 0.0
#   Feature 1: ReLU(1.0 - 0.3) = 0.7
#   Sum: 0.7

# So x_minus_p and p_minus_x are the same! This explains why alpha/beta don't affect result

# Verify manual calculation
print("\nManual calculation:")
print(f"Feature 0: x=1.0, p=0.3")
print(f"  Intersection: 1.0 * 0.3 = 0.3")
print(f"  x-p: ReLU(1.0 - 0.3) = 0.7")
print(f"  p-x: ReLU(0.3 - 1.0) = 0.0")
print(f"Feature 1: x=0.3, p=1.0")
print(f"  Intersection: 0.3 * 1.0 = 0.3")
print(f"  x-p: ReLU(0.3 - 1.0) = 0.0")
print(f"  p-x: ReLU(1.0 - 0.3) = 0.7")
print(f"Total intersection: 0.3 + 0.3 = 0.6")
print(f"Total x-p: 0.7 + 0.0 = 0.7")
print(f"Total p-x: 0.0 + 0.7 = 0.7")

# Test similarity
sim_alpha_high = tversky_similarity(
    x,
    prototypes,
    features,
    alpha=0.9,
    beta=0.1,
    theta=1e-7,
    difference_reduction="substractmatch",
)

sim_beta_high = tversky_similarity(
    x,
    prototypes,
    features,
    alpha=0.1,
    beta=0.9,
    theta=1e-7,
    difference_reduction="substractmatch",
)

print(f"\nSimilarity (α=0.9, β=0.1): {sim_alpha_high}")
print(f"Similarity (α=0.1, β=0.9): {sim_beta_high}")

# Manual computation
theta = 1e-7
num = intersection + theta
denom1 = intersection + 0.9 * x_minus_p + 0.1 * p_minus_x + theta
denom2 = intersection + 0.1 * x_minus_p + 0.9 * p_minus_x + theta
print(f"\nManual (α=0.9, β=0.1): {num/denom1}")
print(f"Manual (α=0.1, β=0.9): {num/denom2}")

# Since x_minus_p == p_minus_x, denominator is the same!
print(f"\nDenom1 = 0.6 + 0.9*0.7 + 0.1*0.7 = 0.6 + 0.63 + 0.07 = 1.3")
print(f"Denom2 = 0.6 + 0.1*0.7 + 0.9*0.7 = 0.6 + 0.07 + 0.63 = 1.3")
print(f"They're the same!")
