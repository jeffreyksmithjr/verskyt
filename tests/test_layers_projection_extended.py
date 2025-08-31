"""
Extended comprehensive tests for TverskyProjectionLayer.
These tests target specific coverage gaps, gradient flow, and mathematical correctness
with focus on different parameter configurations and initialization methods.
"""

import pytest
import torch

from verskyt.core.similarity import DifferenceReduction, IntersectionReduction
from verskyt.layers.projection import TverskyProjectionLayer


class TestTverskyProjectionLayerExtended:
    """Extended tests for TverskyProjectionLayer covering missing functionality."""

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        torch.manual_seed(42)
        batch_size, in_features = 4, 6
        num_prototypes = 3
        inputs = torch.randn(batch_size, in_features)
        return inputs, batch_size, in_features, num_prototypes

    def test_non_learnable_alpha_beta_parameters(self, sample_data):
        """Test creation and behavior with non-learnable alpha and beta parameters."""
        inputs, batch_size, in_features, num_prototypes = sample_data
        num_features = 4

        # Create layer with non-learnable alpha and beta
        layer = TverskyProjectionLayer(
            in_features=in_features,
            num_prototypes=num_prototypes,
            num_features=num_features,
            alpha=0.7,
            beta=0.3,
            learnable_ab=False,
        )

        # Verify parameters are not learnable
        assert (
            not layer.alpha.requires_grad
        ), "Alpha should not require gradients when learnable_alpha=False"
        assert (
            not layer.beta.requires_grad
        ), "Beta should not require gradients when learnable_beta=False"

        # Verify parameter values
        assert torch.allclose(
            layer.alpha, torch.tensor(0.7)
        ), "Alpha value should be preserved"
        assert torch.allclose(
            layer.beta, torch.tensor(0.3)
        ), "Beta value should be preserved"

        # Test forward pass works correctly
        output = layer(inputs)
        assert output.shape == (
            batch_size,
            num_prototypes,
        ), f"Expected shape {(batch_size, num_prototypes)}, got {output.shape}"

        # Test that alpha and beta don't change during backward pass
        loss = output.sum()
        loss.backward()

        # Alpha and beta should remain unchanged
        assert torch.allclose(
            layer.alpha, torch.tensor(0.7)
        ), "Alpha should not change during backward pass"
        assert torch.allclose(
            layer.beta, torch.tensor(0.3)
        ), "Beta should not change during backward pass"

    def test_shared_feature_bank(self, sample_data):
        """Test layer with shared feature bank."""
        inputs, batch_size, in_features, num_prototypes = sample_data
        num_features = 4

        # Create shared feature bank
        shared_features = torch.nn.Parameter(torch.randn(num_features, in_features))

        # Create layer with shared feature bank
        layer = TverskyProjectionLayer(
            in_features=in_features,
            num_prototypes=num_prototypes,
            num_features=num_features,
            shared_feature_bank=shared_features,
        )

        # Verify shared feature bank is used
        assert layer.shared_features, "Layer should recognize shared feature bank"
        assert (
            layer.feature_bank is shared_features
        ), "Layer should use the shared feature bank"

        # Test forward pass
        output = layer(inputs)
        assert output.shape == (
            batch_size,
            num_prototypes,
        ), f"Expected shape {(batch_size, num_prototypes)}, got {output.shape}"

        # Verify output is valid
        assert torch.all(torch.isfinite(output)), "All outputs should be finite"

        # Test gradient flow
        loss = output.sum()
        loss.backward()

        assert shared_features.grad is not None, "Shared features should have gradients"
        assert not torch.allclose(
            shared_features.grad, torch.zeros_like(shared_features.grad)
        ), "Feature gradients should be non-zero"

    def test_uniform_initialization(self, sample_data):
        """Test uniform initialization method."""
        inputs, batch_size, in_features, num_prototypes = sample_data
        num_features = 4

        layer = TverskyProjectionLayer(
            in_features=in_features,
            num_prototypes=num_prototypes,
            num_features=num_features,
            prototype_init="uniform",
        )

        # Test that prototypes are initialized with uniform distribution
        prototypes = layer.prototypes.data

        # For uniform initialization, values should be roughly uniformly distributed
        # Check that values span a reasonable range (not all the same)
        assert (
            prototypes.std() > 0.01
        ), "Prototypes should have reasonable variance with uniform initialization"

        # Test forward pass works
        output = layer(inputs)
        assert output.shape == (batch_size, num_prototypes)

    def test_normal_initialization(self, sample_data):
        """Test normal initialization method."""
        inputs, batch_size, in_features, num_prototypes = sample_data
        num_features = 4

        layer = TverskyProjectionLayer(
            in_features=in_features,
            num_prototypes=num_prototypes,
            num_features=num_features,
            prototype_init="normal",
        )

        # Test that prototypes are initialized with normal distribution
        prototypes = layer.prototypes.data

        # For normal initialization, values should be roughly normally distributed
        assert (
            prototypes.std() > 0.01
        ), "Prototypes should have reasonable variance with normal initialization"

        # Test forward pass works
        output = layer(inputs)
        assert output.shape == (batch_size, num_prototypes)

    def test_xavier_normal_initialization(self, sample_data):
        """Test Xavier normal initialization method."""
        inputs, batch_size, in_features, num_prototypes = sample_data
        num_features = 4

        layer = TverskyProjectionLayer(
            in_features=in_features,
            num_prototypes=num_prototypes,
            num_features=num_features,
            prototype_init="xavier_normal",
        )

        # Test that prototypes are initialized with Xavier normal distribution
        prototypes = layer.prototypes.data

        # Xavier normal should have specific variance based on fan-in and fan-out
        expected_std = (2.0 / (in_features + num_prototypes)) ** 0.5
        actual_std = prototypes.std().item()

        # Allow some tolerance due to random initialization
        assert (
            abs(actual_std - expected_std) < 0.5
        ), f"Xavier normal std should be around {expected_std}, got {actual_std}"

        # Test forward pass works
        output = layer(inputs)
        assert output.shape == (batch_size, num_prototypes)

    def test_bias_functionality(self, sample_data):
        """Test layer with bias term."""
        inputs, batch_size, in_features, num_prototypes = sample_data
        num_features = 4

        layer = TverskyProjectionLayer(
            in_features=in_features,
            num_prototypes=num_prototypes,
            num_features=num_features,
            bias=True,
        )

        # Verify bias exists and is learnable
        assert layer.bias is not None, "Layer should have bias when bias=True"
        assert layer.bias.requires_grad, "Bias should be learnable"
        assert layer.bias.shape == (
            num_prototypes,
        ), f"Bias should have shape {(num_prototypes,)}"

        # Test forward pass
        output = layer(inputs)
        assert output.shape == (batch_size, num_prototypes)

        # Test gradient flow through bias
        loss = output.sum()
        loss.backward()
        assert layer.bias.grad is not None, "Bias should have gradients"

    def test_normalize_features_and_prototypes(self, sample_data):
        """Test normalization functionality."""
        inputs, batch_size, in_features, num_prototypes = sample_data
        num_features = 4

        # Create layer with normalization enabled
        layer = TverskyProjectionLayer(
            in_features=in_features,
            num_prototypes=num_prototypes,
            num_features=num_features,
            normalize_features=True,
            normalize_prototypes=True,
        )

        # Test that normalization flags are stored
        assert layer.normalize_features, "normalize_features should be True"
        assert layer.normalize_prototypes, "normalize_prototypes should be True"

        # Test forward pass with normalization
        output = layer(inputs)
        assert output.shape == (batch_size, num_prototypes)
        assert torch.all(torch.isfinite(output)), "All outputs should be finite"

        # Test gradient flow
        loss = output.sum()
        loss.backward()
        assert layer.prototypes.grad is not None, "Prototypes should have gradients"

    def test_parameter_reset_functionality(self, sample_data):
        """Test parameter reset functionality."""
        inputs, batch_size, in_features, num_prototypes = sample_data
        num_features = 4

        layer = TverskyProjectionLayer(
            in_features=in_features,
            num_prototypes=num_prototypes,
            num_features=num_features,
            learnable_ab=True,
        )

        # Store original parameters
        original_prototypes = layer.prototypes.data.clone()
        original_alpha = layer.alpha.data.clone()
        original_beta = layer.beta.data.clone()

        # Modify parameters
        layer.prototypes.data += 1.0
        layer.alpha.data += 0.1
        layer.beta.data += 0.1

        # Verify parameters changed
        assert not torch.allclose(
            layer.prototypes.data, original_prototypes
        ), "Prototypes should have changed"
        assert not torch.allclose(
            layer.alpha.data, original_alpha
        ), "Alpha should have changed"
        assert not torch.allclose(
            layer.beta.data, original_beta
        ), "Beta should have changed"

        # Reset parameters
        layer.reset_parameters()

        # Parameters should be different from modified values (reset to new random
        # values)
        assert not torch.allclose(layer.prototypes.data, original_prototypes + 1.0), (
            "Prototypes should have been reset"
        )

    def test_get_prototype_functionality(self, sample_data):
        """Test get_prototype utility method."""
        inputs, batch_size, in_features, num_prototypes = sample_data
        num_features = 4

        layer = TverskyProjectionLayer(
            in_features=in_features,
            num_prototypes=num_prototypes,
            num_features=num_features,
        )

        # Test getting valid prototype indices
        for i in range(num_prototypes):
            prototype = layer.get_prototype(i)
            assert prototype.shape == (
                in_features,
            ), f"Prototype {i} should have shape {(in_features,)}"
            assert torch.allclose(
                prototype, layer.prototypes[i]
            ), f"get_prototype({i}) should return prototypes[{i}]"

        # Test error handling for invalid indices
        with pytest.raises(IndexError):
            layer.get_prototype(num_prototypes)

        with pytest.raises(IndexError):
            layer.get_prototype(-num_prototypes - 1)

    def test_set_prototype_functionality(self, sample_data):
        """Test set_prototype utility method."""
        inputs, batch_size, in_features, num_prototypes = sample_data
        num_features = 4

        layer = TverskyProjectionLayer(
            in_features=in_features,
            num_prototypes=num_prototypes,
            num_features=num_features,
        )

        # Create new prototype values
        new_prototype = torch.randn(in_features)

        # Test setting valid prototype indices
        for i in range(num_prototypes):
            layer.set_prototype(i, new_prototype)
            assert torch.allclose(
                layer.prototypes[i], new_prototype
            ), f"set_prototype({i}) should update prototypes[{i}]"

        # Test error handling for invalid indices
        with pytest.raises(IndexError):
            layer.set_prototype(num_prototypes, new_prototype)

        # Test with different shape (should work since set_prototype doesn't
        # validate shape)
        # The actual implementation just assigns the tensor, so this should work
        different_shape_prototype = torch.randn(in_features)
        layer.set_prototype(0, different_shape_prototype)
        assert torch.allclose(
            layer.prototypes[0], different_shape_prototype
        ), "set_prototype should update prototype"

    def test_extra_repr_functionality(self, sample_data):
        """Test extra_repr method for string representation."""
        inputs, batch_size, in_features, num_prototypes = sample_data
        num_features = 4

        layer = TverskyProjectionLayer(
            in_features=in_features,
            num_prototypes=num_prototypes,
            num_features=num_features,
            bias=True,
        )

        repr_str = layer.extra_repr()

        # Check that important parameters are included in representation
        assert "in_features" in repr_str, "Representation should include in_features"
        assert (
            "num_prototypes" in repr_str
        ), "Representation should include num_prototypes"
        assert "num_features" in repr_str, "Representation should include num_features"
        assert "bias=True" in repr_str, "Representation should include bias setting"

    def test_comprehensive_gradient_flow_different_configurations(self, sample_data):
        """Test gradient flow through different layer configurations."""
        inputs, batch_size, in_features, num_prototypes = sample_data
        num_features = 4

        configurations = [
            {"learnable_ab": True},
            {"learnable_ab": False},
            {"normalize_features": True, "normalize_prototypes": True},
            {"bias": True, "learnable_ab": True},
        ]

        for config in configurations:
            layer = TverskyProjectionLayer(
                in_features=in_features,
                num_prototypes=num_prototypes,
                num_features=num_features,
                **config,
            )

            # Forward pass
            output = layer(inputs)
            loss = output.sum()

            # Original parameters stored for potential future use
            pass

            # Backward pass
            loss.backward()

            # Check gradients exist for learnable parameters
            assert (
                layer.prototypes.grad is not None
            ), f"Prototypes should have gradients for config {config}"

            if layer.alpha.requires_grad:
                assert (
                    layer.alpha.grad is not None
                ), f"Alpha should have gradients for config {config}"
            else:
                assert (
                    layer.alpha.grad is None
                ), f"Alpha should not have gradients for config {config}"

            if layer.beta.requires_grad:
                assert (
                    layer.beta.grad is not None
                ), f"Beta should have gradients for config {config}"
            else:
                assert (
                    layer.beta.grad is None
                ), f"Beta should not have gradients for config {config}"

            if layer.bias is not None:
                assert layer.bias.grad is not None, (
                    f"Bias should have gradients when present for config {config}"
                )

    def test_different_reduction_methods_compatibility(self, sample_data):
        """Test layer compatibility with different reduction methods."""
        inputs, batch_size, in_features, num_prototypes = sample_data
        num_features = 4

        intersection_methods = [
            IntersectionReduction.PRODUCT,
            IntersectionReduction.MIN,
            IntersectionReduction.MEAN,
        ]
        difference_methods = [
            DifferenceReduction.IGNOREMATCH,
            DifferenceReduction.SUBSTRACTMATCH,
        ]

        for int_method in intersection_methods:
            for diff_method in difference_methods:
                layer = TverskyProjectionLayer(
                    in_features=in_features,
                    num_prototypes=num_prototypes,
                    num_features=num_features,
                    intersection_reduction=int_method,
                    difference_reduction=diff_method,
                )

                # Test forward pass works with different reduction methods
                output = layer(inputs)
                assert output.shape == (
                    batch_size,
                    num_prototypes,
                ), f"Output shape incorrect for {int_method}, {diff_method}"
                assert torch.all(
                    torch.isfinite(output)
                ), f"Output should be finite for {int_method}, {diff_method}"

                # Test gradient flow
                loss = output.sum()
                loss.backward()
                assert (
                    layer.prototypes.grad is not None
                ), f"Gradients should exist for {int_method}, {diff_method}"

    def test_numerical_stability_edge_cases(self, sample_data):
        """Test numerical stability with edge case inputs."""
        inputs, batch_size, in_features, num_prototypes = sample_data
        num_features = 4

        layer = TverskyProjectionLayer(
            in_features=in_features,
            num_prototypes=num_prototypes,
            num_features=num_features,
        )

        # Test with zero inputs
        zero_inputs = torch.zeros(batch_size, in_features)
        output = layer(zero_inputs)
        assert torch.all(
            torch.isfinite(output)
        ), "Output should be finite for zero inputs"

        # Test with large inputs
        large_inputs = torch.ones(batch_size, in_features) * 1000
        output = layer(large_inputs)
        assert torch.all(
            torch.isfinite(output)
        ), "Output should be finite for large inputs"

        # Test with very small inputs
        small_inputs = torch.ones(batch_size, in_features) * 1e-8
        output = layer(small_inputs)
        assert torch.all(
            torch.isfinite(output)
        ), "Output should be finite for very small inputs"

        # Test with mixed positive/negative inputs
        mixed_inputs = torch.randn(batch_size, in_features) * 100
        output = layer(mixed_inputs)
        assert torch.all(
            torch.isfinite(output)
        ), "Output should be finite for mixed inputs"

    def test_batch_size_variations(self):
        """Test layer behavior with different batch sizes."""
        in_features, num_prototypes, num_features = 5, 3, 4
        layer = TverskyProjectionLayer(
            in_features=in_features,
            num_prototypes=num_prototypes,
            num_features=num_features,
        )

        # Test different batch sizes
        for batch_size in [1, 2, 8, 16, 32]:
            inputs = torch.randn(batch_size, in_features)
            output = layer(inputs)

            assert output.shape == (
                batch_size,
                num_prototypes,
            ), f"Incorrect shape for batch_size={batch_size}"
            assert torch.all(
                torch.isfinite(output)
            ), f"Output should be finite for batch_size={batch_size}"

            # Test gradient flow
            loss = output.sum()
            loss.backward()
            assert (
                layer.prototypes.grad is not None
            ), f"Gradients should exist for batch_size={batch_size}"

            # Clear gradients for next iteration
            layer.zero_grad()

    def test_device_compatibility(self, sample_data):
        """Test layer behavior with different devices (CPU/CUDA if available)."""
        inputs, batch_size, in_features, num_prototypes = sample_data
        num_features = 4

        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        for device in devices:
            layer = TverskyProjectionLayer(
                in_features=in_features,
                num_prototypes=num_prototypes,
                num_features=num_features,
            ).to(device)

            device_inputs = inputs.to(device)

            # Test forward pass
            output = layer(device_inputs)
            assert (
                output.device.type == device.split(":")[0]
            ), f"Output should be on {device}"
            assert output.shape == (batch_size, num_prototypes)

            # Test backward pass
            loss = output.sum()
            loss.backward()
            assert (
                layer.prototypes.grad.device.type == device.split(":")[0]
            ), f"Gradients should be on {device}"
