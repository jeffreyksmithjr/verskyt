"""
Visualization tools for Tversky Neural Networks.

This module provides functions for visualizing and interpreting learned prototypes
and features in TNNs. The functions are designed to make abstract concepts of
"prototypes" and "features" tangible and visible for research analysis.
"""

from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Set a consistent, professional plotting style
sns.set_theme(style="whitegrid", context="notebook")


def plot_prototype_space(
    prototypes: torch.Tensor,
    prototype_labels: List[str],
    features: Optional[torch.Tensor] = None,
    feature_labels: Optional[List[str]] = None,
    reduction_method: str = "pca",
    title: str = "Learned Prototype Space",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Visualizes high-dimensional prototypes and features in a 2D space.

    This function uses dimensionality reduction to project high-dimensional
    prototype and feature vectors into 2D space for visualization. This helps
    researchers understand the conceptual relationships the model has learned.

    Args:
        prototypes (torch.Tensor): The learned prototype vectors of shape
            `[num_prototypes, embedding_dim]`.
        prototype_labels (List[str]): A list of names for each prototype.
        features (Optional[torch.Tensor]): Optional feature vectors to plot,
            e.g., from a grounded feature bank. Shape `[num_features, embedding_dim]`.
        feature_labels (Optional[List[str]]): Optional names for each feature vector.
        reduction_method (str): 'pca' or 'tsne' for dimensionality reduction.
            Defaults to 'pca'.
        title (str): The title of the plot. Defaults to "Learned Prototype Space".
        ax (Optional[plt.Axes]): A matplotlib axes object to plot on. If None,
            a new figure and axes are created.

    Returns:
        plt.Axes: The matplotlib axes object with the plot.

    Raises:
        ValueError: If reduction_method is not 'pca' or 'tsne'.

    Note:
        This visualization is particularly useful for understanding the conceptual
        structure learned by TNNs, as described in Doumbouya et al. (2025).
        PCA preserves global structure while t-SNE is better for local clustering.

    Example:
        >>> prototypes = torch.randn(3, 128)
        >>> labels = ["Low-Risk", "Medium-Risk", "High-Risk"]
        >>> ax = plot_prototype_space(prototypes, labels)
        >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Detach tensors and move to CPU for scikit-learn compatibility
    prototypes_np = prototypes.detach().cpu().numpy()

    # Combine prototypes and features for a single transformation
    combined_vectors = [prototypes_np]
    if features is not None:
        features_np = features.detach().cpu().numpy()
        combined_vectors.append(features_np)

    data_matrix = np.vstack(combined_vectors)

    # Perform dimensionality reduction
    if reduction_method == "pca":
        reducer = PCA(n_components=2)
    elif reduction_method == "tsne":
        reducer = TSNE(n_components=2, perplexity=max(1, len(data_matrix) - 2))
    else:
        raise ValueError("reduction_method must be 'pca' or 'tsne'")

    transformed_data = reducer.fit_transform(data_matrix)

    # Split back into prototypes and features
    transformed_prototypes = transformed_data[: len(prototypes_np)]

    # Plot prototypes as primary points
    sns.scatterplot(
        x=transformed_prototypes[:, 0],
        y=transformed_prototypes[:, 1],
        ax=ax,
        s=150,
        label="Prototypes",
        legend=False,
        marker="o",
    )
    for i, label in enumerate(prototype_labels):
        ax.text(
            transformed_prototypes[i, 0] + 0.02,
            transformed_prototypes[i, 1],
            label,
            fontsize=12,
        )

    # Optionally, plot features as vectors or points
    if features is not None:
        transformed_features = transformed_data[len(prototypes_np) :]
        # Plot features as arrows from the origin
        for i, label in enumerate(feature_labels or []):
            ax.arrow(
                0,
                0,
                transformed_features[i, 0],
                transformed_features[i, 1],
                head_width=0.05,
                head_length=0.1,
                fc="gray",
                ec="gray",
                alpha=0.6,
            )
            ax.text(
                transformed_features[i, 0] * 1.1,
                transformed_features[i, 1] * 1.1,
                label,
                fontsize=10,
                color="gray",
                ha="center",
            )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    return ax


def visualize_prototypes_as_data(
    encoder: torch.nn.Module,
    prototypes: torch.Tensor,
    prototype_labels: List[str],
    dataloader: torch.utils.data.DataLoader,
    top_k: int = 5,
    device: Optional[Union[str, torch.device]] = None,
) -> plt.Figure:
    """
    Visualizes prototypes by showing the top_k most similar data samples.

    This function provides the most intuitive form of interpretation: showing what
    a prototype "looks like" by finding the real data points that are most similar
    to it. This approach is more general than data-domain specification and doesn't
    require retraining.

    Args:
        encoder (torch.nn.Module): The part of the model that produces the embeddings
            (i.e., the layers before the TverskyProjectionLayer).
        prototypes (torch.Tensor): The learned prototype vectors of shape
            `[num_prototypes, embedding_dim]`.
        prototype_labels (List[str]): A list of names for each prototype.
        dataloader (torch.utils.data.DataLoader): A dataloader for the dataset
            (preferably the training set) with `shuffle=False`.
        top_k (int): The number of data samples to show for each prototype.
            Defaults to 5.
        device (Optional[Union[str, torch.device]]): The device to run computations on.
            If None, uses the same device as prototypes.

    Returns:
        plt.Figure: The matplotlib figure containing the visualization.

    Note:
        This function uses cosine similarity to find the most similar data samples
        to each prototype. The visualization assumes image data with channel-first
        format (C, H, W) and converts to channel-last for display.

    Example:
        >>> # Assuming 'model' is a trained TNN with encoder component
        >>> fig = visualize_prototypes_as_data(
        ...     encoder=model.encoder,
        ...     prototypes=model.tnn_layer.prototypes,
        ...     prototype_labels=["Class 0", "Class 1"],
        ...     dataloader=train_loader,
        ...     top_k=3
        ... )
        >>> plt.show()
    """
    if device is None:
        device = prototypes.device

    encoder.eval()
    encoder.to(device)

    # 1. Encode all data points and store their embeddings and original data
    all_embeddings = []
    all_data = []
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            embeddings = encoder(data)
            all_embeddings.append(embeddings.cpu())
            all_data.append(data.cpu())

    all_embeddings = torch.cat(all_embeddings)
    all_data = torch.cat(all_data)

    # 2. For each prototype, find the top_k most similar data embeddings
    prototypes_cpu = prototypes.detach().cpu()

    # Normalize for cosine similarity
    all_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)
    prototypes_cpu = torch.nn.functional.normalize(prototypes_cpu, p=2, dim=1)

    similarity_matrix = torch.matmul(prototypes_cpu, all_embeddings.T)
    top_k_indices = torch.topk(similarity_matrix, k=top_k, dim=1).indices

    # 3. Create the plot
    num_prototypes = len(prototype_labels)
    fig, axes = plt.subplots(
        num_prototypes, top_k, figsize=(top_k * 2, num_prototypes * 2)
    )
    fig.suptitle("Data Samples Most Similar to Each Prototype", fontsize=16)

    # Handle single prototype case
    if num_prototypes == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_prototypes):
        axes[i, 0].set_ylabel(
            prototype_labels[i], rotation=0, labelpad=40, ha="right", fontsize=12
        )
        for j in range(top_k):
            ax = axes[i, j]
            idx = top_k_indices[i, j]
            image = all_data[idx]

            # Handle different data types
            if image.dim() >= 2:
                # Image data - assuming channel-first format if 3D
                if image.dim() == 3:
                    image = image.permute(1, 2, 0)  # C, H, W -> H, W, C
                ax.imshow(image.squeeze(), cmap="gray")
                ax.axis("off")
            else:
                # 1D data - display as text or scatter plot
                if image.dim() == 1 and len(image) <= 10:
                    # Small vector - display as text
                    ax.text(
                        0.5,
                        0.5,
                        f"{image.numpy()}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        transform=ax.transAxes,
                    )
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                else:
                    # Larger vector - display as bar plot
                    ax.bar(range(len(image)), image.numpy())
                ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
