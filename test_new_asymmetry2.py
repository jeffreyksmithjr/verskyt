import torch
from verskyt.core.similarity import tversky_similarity

# Create an asymmetric test case where x and prototypes have different profiles
# that will lead to different x-p and p-x values
x = torch.tensor([[1.0, 0.0]])  # Strong in first feature, none in second
prototypes = torch.tensor([[0.2, 0.8]])  # Weak in first, strong in second
features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # Identity features

print(f"x: {x}")
print(f"prototypes: {prototypes}")

# Compute memberships manually
x_membership = torch.einsum('bi,fi->bf', x, features)  # [1.0, 0.0]
p_membership = torch.einsum('pi,fi->pf', prototypes, features)  # [0.2, 0.8]

print(f"x_membership: {x_membership}")
print(f"p_membership: {p_membership}")

# Manual difference calculation:
# x-p differences: ReLU(1.0-0.2) + ReLU(0.0-0.8) = 0.8 + 0.0 = 0.8
# p-x differences: ReLU(0.2-1.0) + ReLU(0.8-0.0) = 0.0 + 0.8 = 0.8

# Hmm, still symmetric! Let me try a different case
x = torch.tensor([[1.0, 0.1]])  # Strong in first, weak in second
prototypes = torch.tensor([[0.1, 1.0]])  # Weak in first, strong in second
features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # Identity features

print(f"\nNew test case:")
print(f"x: {x}")
print(f"prototypes: {prototypes}")

x_membership = torch.einsum('bi,fi->bf', x, features)  # [1.0, 0.1]
p_membership = torch.einsum('pi,fi->pf', prototypes, features)  # [0.1, 1.0]

print(f"x_membership: {x_membership}")
print(f"p_membership: {p_membership}")

# x-p differences: ReLU(1.0-0.1) + ReLU(0.1-1.0) = 0.9 + 0.0 = 0.9
# p-x differences: ReLU(0.1-1.0) + ReLU(1.0-0.1) = 0.0 + 0.9 = 0.9
# Still symmetric!

# Let me try a truly asymmetric case with different magnitudes
x = torch.tensor([[1.0, 0.2]])  
prototypes = torch.tensor([[0.3, 0.8]])  
features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

print(f"\nAsymmetric test case:")
print(f"x: {x}")
print(f"prototypes: {prototypes}")

x_membership = torch.einsum('bi,fi->bf', x, features)  # [1.0, 0.2]
p_membership = torch.einsum('pi,fi->pf', prototypes, features)  # [0.3, 0.8]

print(f"x_membership: {x_membership}")
print(f"p_membership: {p_membership}")

# x-p differences: ReLU(1.0-0.3) + ReLU(0.2-0.8) = 0.7 + 0.0 = 0.7
# p-x differences: ReLU(0.3-1.0) + ReLU(0.8-0.2) = 0.0 + 0.6 = 0.6
# NOW it's asymmetric!

sim_alpha_high = tversky_similarity(
    x, prototypes, features,
    alpha=0.9, beta=0.1, theta=1e-7,
    difference_reduction="substractmatch"
)

sim_beta_high = tversky_similarity(
    x, prototypes, features,
    alpha=0.1, beta=0.9, theta=1e-7,
    difference_reduction="substractmatch"
)

print(f"\nSimilarity (α=0.9, β=0.1): {sim_alpha_high}")
print(f"Similarity (α=0.1, β=0.9): {sim_beta_high}")
print(f"Difference: {abs(sim_alpha_high - sim_beta_high)}")

# Verify with manual calculation
intersection = 1.0 * 0.3 + 0.2 * 0.8  # 0.3 + 0.16 = 0.46
x_minus_p = 0.7  # As calculated above
p_minus_x = 0.6  # As calculated above

theta = 1e-7
num = intersection + theta
denom1 = intersection + 0.9 * x_minus_p + 0.1 * p_minus_x + theta
denom2 = intersection + 0.1 * x_minus_p + 0.9 * p_minus_x + theta

print(f"\nManual calculation:")
print(f"Intersection: {intersection}")
print(f"x-p: {x_minus_p}")
print(f"p-x: {p_minus_x}")
print(f"Sim (α=0.9, β=0.1): {num/denom1}")
print(f"Sim (α=0.1, β=0.9): {num/denom2}")