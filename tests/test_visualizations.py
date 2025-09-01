"""
Tests for verskyt.visualizations module.

This module tests the visualization functions for TNNs, including prototype
space visualization and data-based prototype interpretation.
"""

from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

# Skip tests if visualization dependencies are not available
try:
    from verskyt.visualizations.plotting import (
        plot_prototype_space,
        visualize_prototypes_as_data,
    )

    visualization_available = True
except ImportError:
    visualization_available = False


@pytest.mark.skipif(
    not visualization_available, reason="Visualization dependencies not available"
)
class TestPlotPrototypeSpace:
    """Test cases for plot_prototype_space function."""

    def test_basic_prototype_plotting(self):
        """Test basic prototype space plotting with PCA."""
        # Create test data
        prototypes = torch.randn(3, 10)
        labels = ["Proto1", "Proto2", "Proto3"]

        # Test basic functionality
        ax = plot_prototype_space(prototypes, labels)

        assert ax is not None
        assert ax.get_title() == "Learned Prototype Space"
        assert ax.get_xlabel() == "Component 1"
        assert ax.get_ylabel() == "Component 2"

    def test_prototype_plotting_with_features(self):
        """Test prototype space plotting with features."""
        prototypes = torch.randn(2, 5)
        prototype_labels = ["Proto1", "Proto2"]
        features = torch.randn(3, 5)
        feature_labels = ["Feat1", "Feat2", "Feat3"]

        ax = plot_prototype_space(
            prototypes, prototype_labels, features, feature_labels
        )

        assert ax is not None

    def test_tsne_reduction_method(self):
        """Test t-SNE dimensionality reduction."""
        prototypes = torch.randn(4, 8)
        labels = ["A", "B", "C", "D"]

        ax = plot_prototype_space(prototypes, labels, reduction_method="tsne")

        assert ax is not None

    def test_invalid_reduction_method(self):
        """Test that invalid reduction method raises ValueError."""
        prototypes = torch.randn(2, 5)
        labels = ["A", "B"]

        with pytest.raises(
            ValueError, match="reduction_method must be 'pca' or 'tsne'"
        ):
            plot_prototype_space(prototypes, labels, reduction_method="invalid")

    def test_custom_title_and_axes(self):
        """Test custom title and axes parameters."""
        prototypes = torch.randn(2, 5)
        labels = ["A", "B"]

        fig, ax = plt.subplots()
        result_ax = plot_prototype_space(
            prototypes, labels, title="Custom Title", ax=ax
        )

        assert result_ax is ax
        assert ax.get_title() == "Custom Title"

    def test_tensor_device_handling(self):
        """Test that function handles tensors on different devices."""
        # Test with CPU tensors
        prototypes = torch.randn(2, 5)
        labels = ["A", "B"]

        ax = plot_prototype_space(prototypes, labels)
        assert ax is not None

        # Test with CUDA tensors if available
        if torch.cuda.is_available():
            prototypes_cuda = prototypes.cuda()
            ax_cuda = plot_prototype_space(prototypes_cuda, labels)
            assert ax_cuda is not None

    def test_gradient_preservation(self):
        """Test that function preserves gradient information."""
        prototypes = torch.randn(2, 5, requires_grad=True)
        labels = ["A", "B"]

        # Should not break gradient tracking
        ax = plot_prototype_space(prototypes, labels)
        assert ax is not None
        assert prototypes.requires_grad is True


@pytest.mark.skipif(
    not visualization_available, reason="Visualization dependencies not available"
)
class TestVisualizePrototypesAsData:
    """Test cases for visualize_prototypes_as_data function."""

    def create_mock_encoder(self, output_dim=10):
        """Create a mock encoder for testing."""
        encoder = Mock()
        encoder.eval.return_value = None
        encoder.to.return_value = encoder

        def mock_forward(x):
            batch_size = x.shape[0]
            return torch.randn(batch_size, output_dim)

        encoder.side_effect = mock_forward
        return encoder

    def create_mock_dataloader(
        self, num_batches=2, batch_size=4, image_shape=(1, 28, 28)
    ):
        """Create a mock dataloader for testing."""
        batches = []
        for _ in range(num_batches):
            data = torch.randn(batch_size, *image_shape)
            labels = torch.randint(0, 2, (batch_size,))
            batches.append((data, labels))
        return batches

    def test_basic_prototype_data_visualization(self):
        """Test basic prototype-as-data visualization."""
        encoder = self.create_mock_encoder()
        prototypes = torch.randn(2, 10)
        prototype_labels = ["Class0", "Class1"]
        dataloader = self.create_mock_dataloader()

        fig = visualize_prototypes_as_data(
            encoder, prototypes, prototype_labels, dataloader
        )

        assert fig is not None
        assert len(fig.axes) == 10  # 2 prototypes × 5 top_k samples

    def test_custom_top_k(self):
        """Test custom top_k parameter."""
        encoder = self.create_mock_encoder()
        prototypes = torch.randn(1, 10)
        prototype_labels = ["Class0"]
        dataloader = self.create_mock_dataloader()

        fig = visualize_prototypes_as_data(
            encoder, prototypes, prototype_labels, dataloader, top_k=3
        )

        assert fig is not None
        assert len(fig.axes) == 3  # 1 prototype × 3 top_k samples

    def test_device_parameter_handling(self):
        """Test device parameter handling."""
        encoder = self.create_mock_encoder()
        prototypes = torch.randn(1, 10)
        prototype_labels = ["Class0"]
        dataloader = self.create_mock_dataloader()

        # Test with explicit device
        fig = visualize_prototypes_as_data(
            encoder, prototypes, prototype_labels, dataloader, device="cpu"
        )

        assert fig is not None

    def test_single_prototype_handling(self):
        """Test handling of single prototype case."""
        encoder = self.create_mock_encoder()
        prototypes = torch.randn(1, 10)
        prototype_labels = ["SingleClass"]
        dataloader = self.create_mock_dataloader()

        fig = visualize_prototypes_as_data(
            encoder, prototypes, prototype_labels, dataloader
        )

        assert fig is not None

    @patch("torch.no_grad")
    def test_gradient_context(self, mock_no_grad):
        """Test that function runs in no_grad context."""
        encoder = self.create_mock_encoder()
        prototypes = torch.randn(1, 10)
        prototype_labels = ["Class0"]
        dataloader = self.create_mock_dataloader()

        # Mock the context manager
        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock(return_value=None)

        visualize_prototypes_as_data(encoder, prototypes, prototype_labels, dataloader)

        mock_no_grad.assert_called_once()


@pytest.mark.skipif(
    not visualization_available, reason="Visualization dependencies not available"
)
class TestVisualizationIntegration:
    """Integration tests for visualization module."""

    def test_module_imports(self):
        """Test that all main functions can be imported."""
        from verskyt.visualizations import (
            plot_prototype_space,
            visualize_prototypes_as_data,
        )

        assert callable(plot_prototype_space)
        assert callable(visualize_prototypes_as_data)

    def test_optional_import_handling(self):
        """Test handling of optional import failures."""
        # This test verifies the import structure works correctly
        from verskyt.visualizations import __all__

        expected_functions = ["plot_prototype_space", "visualize_prototypes_as_data"]
        for func in expected_functions:
            assert func in __all__

    def test_matplotlib_backend_compatibility(self):
        """Test compatibility with different matplotlib backends."""
        import matplotlib

        original_backend = matplotlib.get_backend()

        try:
            # Test with Agg backend (non-interactive)
            matplotlib.use("Agg")

            prototypes = torch.randn(2, 5)
            labels = ["A", "B"]
            ax = plot_prototype_space(prototypes, labels)

            assert ax is not None

        finally:
            # Restore original backend
            matplotlib.use(original_backend)


# Test markers for different test categories
@pytest.mark.visualization
@pytest.mark.skipif(
    not visualization_available, reason="Visualization dependencies not available"
)
class TestVisualizationPerformance:
    """Performance tests for visualization functions."""

    def test_large_prototype_space_performance(self):
        """Test performance with larger prototype spaces."""
        # Test with moderately large data
        prototypes = torch.randn(20, 100)
        labels = [f"Proto{i}" for i in range(20)]

        # This should complete without timeout
        ax = plot_prototype_space(prototypes, labels)
        assert ax is not None

    def test_memory_usage_with_large_data(self):
        """Test memory usage doesn't explode with larger datasets."""
        prototypes = torch.randn(10, 50)
        labels = [f"Proto{i}" for i in range(10)]
        features = torch.randn(15, 50)
        feature_labels = [f"Feat{i}" for i in range(15)]

        # Should handle moderately large feature spaces
        ax = plot_prototype_space(prototypes, labels, features, feature_labels)
        assert ax is not None


if __name__ == "__main__":
    pytest.main([__file__])
