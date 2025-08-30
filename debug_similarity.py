"""
Debug script to understand similarity computation issues.
"""

import torch
from verskyt.core.similarity import tversky_similarity

def debug_orthogonal():
    """Debug the orthogonal case."""
    print("=== Debugging Orthogonal Case ===")
    x = torch.tensor([[1.0, 0.0]])
    prototypes = torch.tensor([[0.0, 1.0]])
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    
    print(f"x: {x}")
    print(f"prototypes: {prototypes}")
    print(f"features: {features}")
    
    # Compute feature memberships manually
    x_membership = torch.einsum('bi,fi->bf', x, features)
    p_membership = torch.einsum('pi,fi->pf', prototypes, features)
    
    print(f"x_membership: {x_membership}")  # Should be [[1, 0]]
    print(f"p_membership: {p_membership}")  # Should be [[0, 1]]
    
    # This should show very little intersection
    similarity = tversky_similarity(
        x, prototypes, features,
        alpha=0.5, beta=0.5, theta=1e-7
    )
    print(f"Similarity: {similarity}")
    print()

def debug_asymmetry():
    """Debug the asymmetry case."""
    print("=== Debugging Asymmetry Case ===")
    x = torch.tensor([[1.0, 0.5]])
    prototypes = torch.tensor([[0.5, 1.0]])
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    
    print(f"x: {x}")
    print(f"prototypes: {prototypes}")
    print(f"features: {features}")
    
    # Compute feature memberships
    x_membership = torch.einsum('bi,fi->bf', x, features)
    p_membership = torch.einsum('pi,fi->pf', prototypes, features)
    
    print(f"x_membership: {x_membership}")  # Should be [[1, 0.5]]
    print(f"p_membership: {p_membership}")  # Should be [[0.5, 1]]
    
    # Forward similarity
    sim_forward = tversky_similarity(
        x, prototypes, features,
        alpha=0.8, beta=0.2, theta=1e-7
    )
    
    # Reverse similarity  
    sim_reverse = tversky_similarity(
        prototypes, x, features,
        alpha=0.2, beta=0.8, theta=1e-7
    )
    
    print(f"Forward similarity (α=0.8, β=0.2): {sim_forward}")
    print(f"Reverse similarity (α=0.2, β=0.8): {sim_reverse}")
    print(f"Different? {not torch.allclose(sim_forward, sim_reverse, atol=1e-3)}")
    print()

if __name__ == "__main__":
    debug_orthogonal()
    debug_asymmetry()