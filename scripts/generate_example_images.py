"""
Generate example images for documentation.

This script runs example code and saves the generated plots as high-quality
images for use in documentation. Images are saved to docs/images/examples/
with consistent naming conventions.
"""

import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from verskyt import TverskyProjectionLayer
from verskyt.benchmarks import run_fast_xor_benchmark

# Import visualization functions
try:
    from verskyt.visualizations import plot_prototype_space
    visualization_available = True
except ImportError:
    print("❌ Visualization dependencies not available")
    print("   Install with: pip install verskyt[visualization]")
    sys.exit(1)

# Configuration
IMAGES_DIR = project_root / "docs" / "images" / "examples"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def save_figure(fig_or_plt, filename, dpi=300):
    """Save figure with consistent settings."""
    filepath = IMAGES_DIR / f"{filename}.png"
    if hasattr(fig_or_plt, 'savefig'):  # It's a figure
        fig_or_plt.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                          facecolor='white', edgecolor='none')
    else:  # It's pyplot
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    print(f"✅ Saved: {filepath}")
    plt.close('all')  # Clean up

def generate_visualization_demo_images():
    """Generate images from visualization demo."""
    print("🎨 Generating visualization demo images...")
    
    # Setup: Create and Train a Simple TNN Model (same as visualization_demo.py)
    class SimpleTNN(nn.Module):
        def __init__(self, input_dim=2, hidden_dim=8, output_dim=2, num_prototypes=4):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.tnn_layer = TverskyProjectionLayer(
                in_features=hidden_dim,
                num_prototypes=num_prototypes,
                num_features=16,
                alpha=1.0,
                beta=1.0
            )
            self.output_layer = nn.Linear(num_prototypes, output_dim)

        def forward(self, x):
            encoded = self.encoder(x)
            tnn_out = self.tnn_layer(encoded)
            return self.output_layer(tnn_out)

    # Create synthetic data
    torch.manual_seed(42)
    n_samples = 200
    X = torch.randn(n_samples, 2)
    X[:n_samples//2] += torch.tensor([2.0, 2.0])
    X[n_samples//2:] += torch.tensor([-2.0, -2.0])
    y = torch.cat([torch.zeros(n_samples//2), torch.ones(n_samples//2)]).long()

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train model
    model = SimpleTNN(input_dim=2, hidden_dim=8, output_dim=2, num_prototypes=4)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print("  Training model...")
    model.train()
    for epoch in range(50):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    
    # Generate Image 1: Prototype Space Analysis (PCA/t-SNE)
    prototypes = model.tnn_layer.prototypes.data
    prototype_labels = [f"Prototype {i+1}" for i in range(prototypes.shape[0])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    plot_prototype_space(
        prototypes=prototypes,
        prototype_labels=prototype_labels,
        reduction_method='pca',
        title="Prototype Space (PCA)",
        ax=ax1
    )
    
    if prototypes.shape[0] >= 3:
        plot_prototype_space(
            prototypes=prototypes,
            prototype_labels=prototype_labels,
            reduction_method='tsne',
            title="Prototype Space (t-SNE)",
            ax=ax2
        )
    else:
        ax2.text(0.5, 0.5, "t-SNE requires more prototypes", 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Prototype Space (t-SNE)")

    plt.tight_layout()
    save_figure(fig, "visualization_demo_prototype_space")

    # Generate Image 2: Data Points Colored by Prototype Similarity
    model.eval()
    with torch.no_grad():
        all_embeddings = []
        all_data = []
        all_labels = []
        
        for batch_x, batch_y in DataLoader(dataset, batch_size=32, shuffle=False):
            encoded = model.encoder(batch_x)
            all_embeddings.append(encoded)
            all_data.append(batch_x)
            all_labels.append(batch_y)
        
        all_embeddings = torch.cat(all_embeddings)
        all_data = torch.cat(all_data)
        all_labels = torch.cat(all_labels)
        
        similarities = model.tnn_layer(all_embeddings)
        most_similar_prototypes = torch.argmax(similarities, dim=1)

    plt.figure(figsize=(10, 8))
    colors = ['red', 'blue', 'green', 'orange']
    for i in range(len(prototype_labels)):
        mask = most_similar_prototypes == i
        if mask.sum() > 0:
            plt.scatter(all_data[mask, 0], all_data[mask, 1], 
                       c=colors[i], label=f'Most similar to {prototype_labels[i]}',
                       alpha=0.6, s=50)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2') 
    plt.title('Data Points Colored by Most Similar Prototype')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_figure(plt, "visualization_demo_data_clustering")

    # Generate Image 3: Prototype-Feature Relationship Analysis
    feature_dim = prototypes.shape[1]
    synthetic_features = torch.randn(6, feature_dim)
    feature_labels = [f"Feature_{chr(65+i)}" for i in range(6)]

    ax = plot_prototype_space(
        prototypes=prototypes,
        prototype_labels=prototype_labels,
        features=synthetic_features,
        feature_labels=feature_labels,
        title="Prototype-Feature Relationship Analysis",
        reduction_method="pca",
    )
    save_figure(plt, "visualization_demo_prototype_features")

    print(f"✅ Generated visualization demo images")

def generate_xor_visualization():
    """Generate XOR prototype visualization."""
    print("⚡ Generating XOR visualization...")
    
    try:
        # Create simple XOR model
        xor_model = TverskyProjectionLayer(
            in_features=2,
            num_prototypes=2,
            num_features=4,
            alpha=0.5,
            beta=0.5,
        )

        # Quick training on XOR data
        xor_inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        xor_targets = torch.tensor([0, 1, 1, 0]).float()

        optimizer = torch.optim.Adam(xor_model.parameters(), lr=0.1)

        for epoch in range(100):
            optimizer.zero_grad()
            outputs = xor_model(xor_inputs)
            predictions = torch.softmax(outputs, dim=1)[:, 1]
            loss = torch.nn.functional.binary_cross_entropy(predictions, xor_targets)
            loss.backward()
            optimizer.step()

        # Visualize learned XOR prototypes
        xor_prototypes = xor_model.prototypes.data
        xor_labels = [f"XOR_Class_{i}" for i in range(xor_prototypes.shape[0])]

        plot_prototype_space(
            prototypes=xor_prototypes,
            prototype_labels=xor_labels,
            title="XOR Problem: Learned Prototype Space",
        )
        save_figure(plt, "visualization_demo_xor_prototypes")
        
        print("✅ Generated XOR visualization")

    except Exception as e:
        print(f"⚠️ XOR visualization failed: {e}")

def main():
    """Generate all example images."""
    print("🚀 Generating example images for documentation...")
    print(f"📁 Saving to: {IMAGES_DIR}")
    
    if not visualization_available:
        print("❌ Cannot generate images without visualization dependencies")
        return
    
    try:
        generate_visualization_demo_images()
        generate_xor_visualization()
        
        print(f"\n✅ All images generated successfully!")
        print(f"📁 Images saved to: {IMAGES_DIR}")
        print("\n📋 Generated files:")
        for img_file in sorted(IMAGES_DIR.glob("*.png")):
            print(f"   - {img_file.name}")
            
    except Exception as e:
        print(f"❌ Error generating images: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()