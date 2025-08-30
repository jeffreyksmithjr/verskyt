"""
Basic functionality tests to verify core implementation works.

These are the most critical tests that must pass before proceeding.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from verskyt.core.similarity import tversky_similarity
from verskyt.layers.projection import (
    TverskyProjectionLayer,
    TverskySimilarityLayer,
)


class TestBasicSimilarity:
    """Basic tests for Tversky similarity function."""

    def test_similarity_shape(self):
        """Test output shape is correct."""
        batch_size = 3
        num_prototypes = 2
        in_features = 4
        num_features = 5

        x = torch.randn(batch_size, in_features)
        prototypes = torch.randn(num_prototypes, in_features)
        features = torch.randn(num_features, in_features)

        similarity = tversky_similarity(x, prototypes, features, alpha=0.5, beta=0.5)

        assert similarity.shape == (batch_size, num_prototypes)

    def test_similarity_range(self):
        """Test similarity values are in valid range [0, 1]."""
        x = torch.randn(2, 3)
        prototypes = torch.randn(2, 3)
        features = torch.randn(4, 3)

        similarity = tversky_similarity(x, prototypes, features, alpha=0.5, beta=0.5)

        assert torch.all(similarity >= 0)
        assert torch.all(similarity <= 1)

    def test_identical_objects(self):
        """Test that identical objects have high similarity."""
        x = torch.tensor([[1.0, 0.0]])
        prototypes = x.clone()  # Identical
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        similarity = tversky_similarity(x, prototypes, features, alpha=0.5, beta=0.5)

        # Should be close to 1 for identical objects
        assert similarity[0, 0] > 0.9


class TestTverskyProjectionLayer:
    """Basic tests for TverskyProjectionLayer."""

    def test_layer_creation(self):
        """Test layer can be created successfully."""
        layer = TverskyProjectionLayer(in_features=10, num_prototypes=5, num_features=8)

        assert layer.in_features == 10
        assert layer.num_prototypes == 5
        assert layer.num_features == 8
        assert layer.prototypes.shape == (5, 10)
        assert layer.feature_bank.shape == (8, 10)

    def test_forward_pass(self):
        """Test forward pass works and has correct output shape."""
        layer = TverskyProjectionLayer(in_features=6, num_prototypes=3, num_features=4)

        batch_size = 5
        x = torch.randn(batch_size, 6)

        output = layer(x)

        assert output.shape == (batch_size, 3)
        assert torch.all(output >= 0)  # Similarities should be non-negative
        assert torch.all(output <= 1)  # And <= 1

    def test_gradient_flow(self):
        """Test that gradients flow through the layer."""
        layer = TverskyProjectionLayer(
            in_features=4, num_prototypes=2, num_features=3, learnable_ab=True
        )

        x = torch.randn(2, 4, requires_grad=True)
        output = layer(x)
        loss = output.sum()

        loss.backward()

        # Check gradients exist and are not NaN
        assert layer.prototypes.grad is not None
        assert layer.feature_bank.grad is not None
        assert layer.alpha.grad is not None
        assert layer.beta.grad is not None
        assert not torch.isnan(layer.prototypes.grad).any()
        assert not torch.isnan(layer.feature_bank.grad).any()


class TestTverskySimilarityLayer:
    """Basic tests for TverskySimilarityLayer."""

    def test_similarity_layer_creation(self):
        """Test similarity layer can be created."""
        layer = TverskySimilarityLayer(in_features=8, num_features=6)

        assert layer.in_features == 8
        assert layer.num_features == 6
        assert layer.feature_bank.shape == (6, 8)

    def test_similarity_forward(self):
        """Test similarity layer forward pass."""
        layer = TverskySimilarityLayer(in_features=5, num_features=4)

        a = torch.randn(3, 5)
        b = torch.randn(3, 5)

        similarity = layer(a, b)

        assert similarity.shape == (3,)
        assert torch.all(similarity >= 0)
        assert torch.all(similarity <= 1)


class TestSimpleXOR:
    """Test that we can at least set up XOR problem (not necessarily solve it yet)."""

    def test_xor_setup(self):
        """Test XOR problem setup - layer should accept XOR inputs."""
        # XOR truth table
        xor_inputs = torch.tensor(
            [
                [0.0, 0.0],  # -> 0
                [0.0, 1.0],  # -> 1
                [1.0, 0.0],  # -> 1
                [1.0, 1.0],  # -> 0
            ]
        )

        layer = TverskyProjectionLayer(
            in_features=2,
            num_prototypes=2,  # Binary classification
            num_features=2,  # Minimal features for XOR
        )

        output = layer(xor_inputs)

        # Should produce valid output
        assert output.shape == (4, 2)
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

    def test_xor_training_step(self):
        """Test that we can take a training step on XOR without errors."""
        xor_inputs = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )
        xor_targets = torch.tensor([0, 1, 1, 0])

        layer = TverskyProjectionLayer(
            in_features=2, num_prototypes=2, num_features=2, learnable_ab=True
        )

        optimizer = optim.Adam(layer.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # One training step
        optimizer.zero_grad()
        output = layer(xor_inputs)
        loss = criterion(output, xor_targets)
        loss.backward()
        optimizer.step()

        # Should complete without errors
        assert not torch.isnan(loss)
        assert loss.item() > 0  # Loss should be positive

    def test_xor_learning_capability(self):
        """Test that single layer TNN can learn on XOR problem (validates trainability and non-linear potential)."""
        # XOR problem setup
        xor_inputs = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 1.0], 
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )
        xor_targets = torch.tensor([0, 1, 1, 0])

        # Single layer TNN
        torch.manual_seed(42)  # For reproducible test
        layer = TverskyProjectionLayer(
            in_features=2, 
            num_prototypes=2, 
            num_features=4,
            learnable_ab=True,
            alpha=0.5,
            beta=0.5,
            feature_init="uniform"
        )

        optimizer = optim.Adam(layer.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()

        # Record initial loss
        with torch.no_grad():
            initial_output = layer(xor_inputs)
            initial_loss = criterion(initial_output, xor_targets).item()
        
        # Train for several epochs
        for epoch in range(200):
            optimizer.zero_grad()
            output = layer(xor_inputs)
            loss = criterion(output, xor_targets)
            loss.backward()
            optimizer.step()

        # Validate that learning occurred
        with torch.no_grad():
            final_output = layer(xor_inputs)
            final_loss = criterion(final_output, xor_targets).item()
            predictions = torch.argmax(final_output, dim=1)
            
            # Check that learning occurred (loss decreased)
            assert final_loss < initial_loss, f"No learning occurred: initial loss {initial_loss:.3f}, final loss {final_loss:.3f}"
            
            # Check that network shows non-trivial behavior (not just predicting one class)
            unique_predictions = len(torch.unique(predictions))
            assert unique_predictions > 1, f"Network collapsed to single prediction class: {predictions}"
            
            # Check that network achieves better than random performance (50% for 2-class)
            correct = (predictions == xor_targets).sum().item()
            accuracy = correct / len(xor_targets)
            assert accuracy > 0.5, f"Accuracy {accuracy:.2f} not better than random (0.5)"


class TestParameterLearning:
    """Test that parameters can be learned."""

    def test_alpha_beta_learning(self):
        """Test that alpha and beta parameters update during training."""
        # Set seed for reproducible test
        torch.manual_seed(42)

        layer = TverskyProjectionLayer(
            in_features=3,
            num_prototypes=2,
            num_features=2,
            learnable_ab=True,
            alpha=0.5,
            beta=0.5,
        )

        # Store initial values
        initial_alpha = layer.alpha.clone().detach()
        initial_beta = layer.beta.clone().detach()

        # Training step with higher learning rate
        x = torch.randn(4, 3)
        targets = torch.randint(0, 2, (4,))
        optimizer = optim.SGD(layer.parameters(), lr=1.0)  # Higher learning rate
        criterion = nn.CrossEntropyLoss()

        # Multiple training steps to ensure parameter changes
        for _ in range(3):
            optimizer.zero_grad()
            output = layer(x)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

        # Parameters should have changed
        assert not torch.equal(
            layer.alpha, initial_alpha
        ), f"Alpha didn't change: {layer.alpha} vs {initial_alpha}"
        assert not torch.equal(
            layer.beta, initial_beta
        ), f"Beta didn't change: {layer.beta} vs {initial_beta}"

    def test_prototype_learning(self):
        """Test that prototypes update during training."""
        layer = TverskyProjectionLayer(in_features=4, num_prototypes=3, num_features=3)

        initial_prototypes = layer.prototypes.clone().detach()

        # Training step
        x = torch.randn(5, 4)
        targets = torch.randint(0, 3, (5,))
        optimizer = optim.SGD(layer.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()

        optimizer.zero_grad()
        output = layer(x)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        # Prototypes should have changed
        assert not torch.equal(layer.prototypes, initial_prototypes)
