"""
Test different reduction methods to see which gives expected results.
"""

import torch
from verskyt.core.similarity import tversky_similarity

def test_methods():
    """Test different intersection and difference methods."""
    # Orthogonal case
    x = torch.tensor([[1.0, 0.0]])
    prototypes = torch.tensor([[0.0, 1.0]])
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    
    methods = [
        ("product", "ignorematch"),
        ("product", "substractmatch"), 
        ("min", "ignorematch"),
        ("min", "substractmatch"),
    ]
    
    print("=== Testing Orthogonal Objects ===")
    print(f"x: {x}")
    print(f"prototypes: {prototypes}")
    print()
    
    for int_method, diff_method in methods:
        similarity = tversky_similarity(
            x, prototypes, features,
            alpha=0.5, beta=0.5, theta=1e-7,
            intersection_reduction=int_method,
            difference_reduction=diff_method
        )
        print(f"{int_method} + {diff_method}: {similarity.item():.4f}")
    
    print("\n=== Testing Identical Objects ===")
    x_identical = torch.tensor([[1.0, 0.0]])
    prototypes_identical = torch.tensor([[1.0, 0.0]])  # Same as x
    
    for int_method, diff_method in methods:
        similarity = tversky_similarity(
            x_identical, prototypes_identical, features,
            alpha=0.5, beta=0.5, theta=1e-7,
            intersection_reduction=int_method,
            difference_reduction=diff_method
        )
        print(f"{int_method} + {diff_method}: {similarity.item():.4f}")

if __name__ == "__main__":
    test_methods()