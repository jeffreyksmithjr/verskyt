"""
Unit tests for core Tversky similarity functions.

Tests verify mathematical correctness against paper specifications.
"""

import torch

from verskyt.core.similarity import (
    compute_feature_membership,
    compute_salience,
    tversky_similarity,
)


class TestFeatureMembership:
    """Test feature membership computation."""

    def test_membership_computation(self):
        """Test that membership is computed as x·f per paper."""
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        membership = compute_feature_membership(x, features)

        # Expected: [[1, 0], [0, 1], [1, 1]]
        expected = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        assert torch.allclose(membership, expected)

    def test_membership_sign(self):
        """Test that negative dot products are handled correctly."""
        x = torch.tensor([[1.0, 0.0]])
        features = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])

        membership = compute_feature_membership(x, features)

        # x·f0 = -1, x·f1 = 1
        expected = torch.tensor([[-1.0, 1.0]])
        assert torch.allclose(membership, expected)


class TestSalience:
    """Test salience computation (Equation 2)."""

    def test_salience_positive_only(self):
        """Test that salience only sums positive memberships."""
        x = torch.tensor([[1.0, 1.0], [-1.0, 1.0]])
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

        salience = compute_salience(x, features)

        # x0: memberships [1, 1, -1] -> salience = 1 + 1 = 2
        # x1: memberships [-1, 1, 1] -> salience = 1 + 1 = 2
        expected = torch.tensor([2.0, 2.0])
        assert torch.allclose(salience, expected)


class TestTverskySimilarity:
    """Test main Tversky similarity function."""

    def test_identical_objects(self):
        """Test that identical objects have maximum similarity."""
        x = torch.tensor([[1.0, 0.0, 0.0]])
        prototypes = torch.tensor([[1.0, 0.0, 0.0]])
        features = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        similarity = tversky_similarity(x, prototypes, features, alpha=0.5, beta=0.5, theta=1e-7)

        # Identical objects should have similarity close to 1
        assert similarity[0, 0] > 0.99

    def test_orthogonal_objects(self):
        """Test that orthogonal objects have low similarity."""
        x = torch.tensor([[1.0, 0.0]])
        prototypes = torch.tensor([[0.0, 1.0]])
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        similarity = tversky_similarity(
            x,
            prototypes,
            features,
            alpha=0.5,
            beta=0.5,
            theta=1e-7,
            difference_reduction="ignorematch",  # Use ignorematch for intuitive orthogonal behavior
        )

        # Orthogonal objects should have low similarity
        assert similarity[0, 0] < 0.5

    def test_asymmetry(self):
        """Test that similarity can be asymmetric with different α and β."""
        # Create case with asymmetric differences to show asymmetry
        x = torch.tensor([[1.0, 0.2]])  # Strong in first feature, weak in second
        prototypes = torch.tensor([[0.3, 0.8]])  # Weak in first feature, strong in second
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # Identity features

        # This creates asymmetric differences:
        # x-p: ReLU(1.0-0.3) + ReLU(0.2-0.8) = 0.7 + 0.0 = 0.7
        # p-x: ReLU(0.3-1.0) + ReLU(0.8-0.2) = 0.0 + 0.6 = 0.6

        # Test asymmetry by varying α,β weights for same comparison
        sim_alpha_high = tversky_similarity(
            x,
            prototypes,
            features,
            alpha=0.9,
            beta=0.1,
            theta=1e-7,
            difference_reduction="substractmatch",  # Use substractmatch for magnitude differences
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

        # With different α,β weights, similarity should differ
        # α weights x's advantages, β weights prototype's advantages
        assert not torch.allclose(sim_alpha_high, sim_beta_high, atol=1e-3)


class TestIntersectionReductions:
    """Test different intersection reduction methods."""

    def setup_method(self):
        """Set up test data."""
        self.x = torch.tensor([[2.0, 3.0]])
        self.prototypes = torch.tensor([[1.0, 2.0]])
        self.features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    def test_product_reduction(self):
        """Test product intersection reduction."""
        similarity = tversky_similarity(
            self.x,
            self.prototypes,
            self.features,
            alpha=0.5,
            beta=0.5,
            theta=1e-7,
            intersection_reduction="product",
        )
        assert similarity.shape == (1, 1)
        assert similarity[0, 0] > 0

    def test_min_reduction(self):
        """Test minimum intersection reduction."""
        similarity = tversky_similarity(
            self.x,
            self.prototypes,
            self.features,
            alpha=0.5,
            beta=0.5,
            theta=1e-7,
            intersection_reduction="min",
        )
        assert similarity.shape == (1, 1)
        assert similarity[0, 0] > 0

    def test_mean_reduction(self):
        """Test mean intersection reduction."""
        similarity = tversky_similarity(
            self.x,
            self.prototypes,
            self.features,
            alpha=0.5,
            beta=0.5,
            theta=1e-7,
            intersection_reduction="mean",
        )
        assert similarity.shape == (1, 1)
        assert similarity[0, 0] > 0


class TestDifferenceReductions:
    """Test different difference reduction methods."""

    def setup_method(self):
        """Set up test data."""
        self.x = torch.tensor([[1.0, 0.5]])
        self.prototypes = torch.tensor([[0.5, 1.0]])
        self.features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    def test_ignorematch_reduction(self):
        """Test ignorematch difference reduction."""
        similarity = tversky_similarity(
            self.x,
            self.prototypes,
            self.features,
            alpha=0.5,
            beta=0.5,
            theta=1e-7,
            difference_reduction="ignorematch",
        )
        assert similarity.shape == (1, 1)
        assert 0 <= similarity[0, 0] <= 1

    def test_substractmatch_reduction(self):
        """Test substractmatch difference reduction."""
        similarity = tversky_similarity(
            self.x,
            self.prototypes,
            self.features,
            alpha=0.5,
            beta=0.5,
            theta=1e-7,
            difference_reduction="substractmatch",
        )
        assert similarity.shape == (1, 1)
        assert 0 <= similarity[0, 0] <= 1


class TestBatchProcessing:
    """Test batch processing capabilities."""

    def test_batch_similarity(self):
        """Test similarity computation with batched inputs."""
        batch_size = 4
        x = torch.randn(batch_size, 5)
        prototypes = torch.randn(3, 5)
        features = torch.randn(10, 5)

        similarity = tversky_similarity(x, prototypes, features, alpha=0.5, beta=0.5)

        assert similarity.shape == (batch_size, 3)
        assert torch.all(similarity >= 0)
        assert torch.all(similarity <= 1)


class TestNumericalStability:
    """Test numerical stability with edge cases."""

    def test_zero_vectors(self):
        """Test with zero vectors."""
        x = torch.zeros(1, 3)
        prototypes = torch.zeros(1, 3)
        features = torch.randn(5, 3)

        similarity = tversky_similarity(x, prototypes, features, alpha=0.5, beta=0.5, theta=1e-7)

        assert not torch.isnan(similarity).any()
        assert not torch.isinf(similarity).any()

    def test_large_values(self):
        """Test with large values."""
        x = torch.randn(2, 3) * 1000
        prototypes = torch.randn(2, 3) * 1000
        features = torch.randn(5, 3) * 1000

        similarity = tversky_similarity(x, prototypes, features, alpha=0.5, beta=0.5, theta=1e-7)

        assert not torch.isnan(similarity).any()
        assert not torch.isinf(similarity).any()
        assert torch.all(similarity >= 0)
        assert torch.all(similarity <= 1)


class TestGradientFlow:
    """Test gradient flow through similarity function."""

    def test_gradient_computation(self):
        """Test that gradients can be computed."""
        x = torch.randn(2, 3, requires_grad=True)
        prototypes = torch.randn(2, 3, requires_grad=True)
        features = torch.randn(5, 3, requires_grad=True)

        similarity = tversky_similarity(x, prototypes, features, alpha=0.5, beta=0.5)

        loss = similarity.sum()
        loss.backward()

        assert x.grad is not None
        assert prototypes.grad is not None
        assert features.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(prototypes.grad).any()
        assert not torch.isnan(features.grad).any()
