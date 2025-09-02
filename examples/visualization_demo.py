"""
Verskyt Visualization Demo: Interpreting Learned Prototypes

This demo showcases the visualization capabilities of Verskyt for interpreting
and understanding Tversky Neural Network prototypes. It demonstrates:

1. Prototype space visualization using dimensionality reduction
2. Data-based prototype interpretation showing similar samples
3. Integration with trained TNN models for research analysis

Note: This example requires visualization dependencies:
    pip install verskyt[visualization]
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import Verskyt components
from verskyt import TverskyProjectionLayer
from verskyt.benchmarks import run_fast_xor_benchmark

# Import visualization functions (requires optional dependencies)
try:
    from verskyt.visualizations import plot_prototype_space

    visualization_available = True
except ImportError:
    print("⚠️  Visualization dependencies not available.")
    print("   Install with: pip install verskyt[visualization]")
    visualization_available = False
    exit(1)

print("🎨 Verskyt Visualization Demo: Interpreting Learned Prototypes")
print("=" * 70)

# =============================================================================
# Setup: Create and Train a Simple TNN Model
# =============================================================================
print("\n🔧 SETUP: Training a Simple TNN Model")
print("-" * 40)


# Create a simple TNN model for demonstration
class SimpleTNN(nn.Module):
    """Simple TNN model with encoder and TverskyProjectionLayer."""

    def __init__(self, input_dim=2, hidden_dim=8, output_dim=2, num_prototypes=4):
        super().__init__()
        # Encoder part (the part before TverskyProjectionLayer)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # TNN layer
        self.tnn_layer = TverskyProjectionLayer(
            in_features=hidden_dim,
            num_prototypes=num_prototypes,
            num_features=16,  # Feature bank size
            alpha=1.0,
            beta=1.0,
        )

        # Output projection layer
        self.output_layer = nn.Linear(num_prototypes, output_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        tnn_out = self.tnn_layer(encoded)
        return self.output_layer(tnn_out)


# Create synthetic data for demonstration
torch.manual_seed(42)
n_samples = 200
X = torch.randn(n_samples, 2)
# Create two clusters for binary classification
X[: n_samples // 2] += torch.tensor([2.0, 2.0])
X[n_samples // 2 :] += torch.tensor([-2.0, -2.0])
y = torch.cat([torch.zeros(n_samples // 2), torch.ones(n_samples // 2)]).long()

# Create data loader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize and train the model
model = SimpleTNN(input_dim=2, hidden_dim=8, output_dim=2, num_prototypes=4)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

print("Training model...")
model.train()
for epoch in range(50):  # Quick training for demo
    total_loss = 0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/50, Loss: {total_loss/len(dataloader):.4f}")

model.eval()
print("✅ Model training completed!")

# =============================================================================
# Visualization 1: Prototype Space Visualization
# =============================================================================
print("\n🎯 VISUALIZATION 1: Prototype Space Analysis")
print("-" * 45)

print("Visualizing learned prototype space using PCA...")

# Extract learned prototypes
prototypes = model.tnn_layer.prototypes.data
prototype_labels = [f"Prototype {i+1}" for i in range(prototypes.shape[0])]

# Create prototype space visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# PCA visualization
plot_prototype_space(
    prototypes=prototypes,
    prototype_labels=prototype_labels,
    reduction_method="pca",
    title="Prototype Space (PCA)",
    ax=ax1,
)

# t-SNE visualization (if enough prototypes)
if prototypes.shape[0] >= 3:
    plot_prototype_space(
        prototypes=prototypes,
        prototype_labels=prototype_labels,
        reduction_method="tsne",
        title="Prototype Space (t-SNE)",
        ax=ax2,
    )
else:
    ax2.text(
        0.5,
        0.5,
        "t-SNE requires more prototypes",
        ha="center",
        va="center",
        transform=ax2.transAxes,
    )
    ax2.set_title("Prototype Space (t-SNE)")

plt.tight_layout()
plt.show()

print("📊 Prototype space visualization shows the conceptual relationships")
print("   learned by the TNN in reduced dimensionality.")

# =============================================================================
# Visualization 2: Data Points Colored by Prototype Similarity
# =============================================================================
print("\n🔍 VISUALIZATION 2: Data Points Colored by Prototype Similarity")
print("-" * 55)

print("Coloring data points by their most similar prototype...")

# Encode all data points and find most similar prototypes
model.eval()
with torch.no_grad():
    # Get all data points
    all_embeddings = []
    all_data = []
    all_labels = []

    for batch_x, batch_y in DataLoader(dataset, batch_size=32, shuffle=False):
        encoded = model.encoder(batch_x)
        tnn_out = model.tnn_layer(encoded)  # This gives similarity to prototypes
        all_embeddings.append(encoded)
        all_data.append(batch_x)
        all_labels.append(batch_y)

    all_embeddings = torch.cat(all_embeddings)
    all_data = torch.cat(all_data)
    all_labels = torch.cat(all_labels)

    # Find which prototype each data point is most similar to
    # Use the TNN layer output (similarity to prototypes)
    similarities = model.tnn_layer(all_embeddings)
    most_similar_prototypes = torch.argmax(similarities, dim=1)

# Create scatter plot colored by most similar prototype
plt.figure(figsize=(10, 8))
colors = ["red", "blue", "green", "orange"]
for i in range(len(prototype_labels)):
    mask = most_similar_prototypes == i
    if mask.sum() > 0:
        plt.scatter(
            all_data[mask, 0],
            all_data[mask, 1],
            c=colors[i],
            label=f"Most similar to {prototype_labels[i]}",
            alpha=0.6,
            s=50,
        )

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Data Points Colored by Most Similar Prototype")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(
    "📊 This visualization shows how data points cluster around different prototypes,"
)
print("   revealing the decision boundaries learned by the TNN.")

# =============================================================================
# Advanced Analysis: Prototype-Feature Relationships
# =============================================================================
print("\n🧬 ADVANCED: Prototype-Feature Analysis")
print("-" * 40)

print("Analyzing relationships between prototypes and synthetic features...")

# Create synthetic feature vectors for demonstration
feature_dim = prototypes.shape[1]
synthetic_features = torch.randn(6, feature_dim)
feature_labels = [f"Feature_{chr(65+i)}" for i in range(6)]  # A, B, C, D, E, F

# Visualize prototypes with features (function creates its own figure)
ax = plot_prototype_space(
    prototypes=prototypes,
    prototype_labels=prototype_labels,
    features=synthetic_features,
    feature_labels=feature_labels,
    title="Prototype-Feature Relationship Analysis",
    reduction_method="pca",
)
plt.show()

print("🎯 This analysis shows how learned prototypes relate to conceptual")
print("   features in the embedding space, useful for interpretability research.")

# =============================================================================
# Research Insights and Next Steps
# =============================================================================
print("\n💡 RESEARCH INSIGHTS & NEXT STEPS")
print("-" * 35)

print("Key visualization insights:")
print("• Prototype Space Analysis reveals conceptual clustering")
print("• Data-based interpretation shows prototype 'meaning' through examples")
print("• Feature relationships help understand learned representations")
print()
print("For advanced research:")
print("• Use these visualizations to validate prototype learning")
print("• Compare different TNN configurations visually")
print("• Analyze prototype stability across training runs")
print("• Study feature grounding and intervention effects")
print()
print("📚 See the research tutorial for more advanced TNN capabilities!")

# =============================================================================
# Performance Comparison Demo
# =============================================================================
print("\n⚡ BONUS: XOR Benchmark with Visualization")
print("-" * 40)

print("Running XOR benchmark and visualizing learned prototypes...")

# Run a quick XOR benchmark using the standalone function

results, analysis = run_fast_xor_benchmark()

print("XOR Benchmark Results:")
print(f"• Overall Convergence Rate: {analysis['overall_convergence_rate']:.2%}")
print(f"• Total Experimental Runs: {len(results)}")

# Note: XOR benchmark doesn't store the trained model for visualization
# But we can create a simple XOR TNN model for demonstration
print("Creating simple XOR model for prototype visualization...")

try:
    # Create a simple XOR model using TverskyProjectionLayer
    xor_model = TverskyProjectionLayer(
        in_features=2,
        num_prototypes=2,  # XOR has 2 output classes
        num_features=4,  # Small feature bank for XOR
        alpha=0.5,
        beta=0.5,
    )

    # Quick training on XOR data
    xor_inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    xor_targets = torch.tensor([0, 1, 1, 0]).float()

    optimizer = torch.optim.Adam(xor_model.parameters(), lr=0.1)

    for epoch in range(100):  # Quick training
        optimizer.zero_grad()
        outputs = xor_model(xor_inputs)
        # Convert to binary classification loss
        predictions = torch.softmax(outputs, dim=1)[:, 1]  # Take class 1 probabilities
        loss = torch.nn.functional.binary_cross_entropy(predictions, xor_targets)
        loss.backward()
        optimizer.step()

    # Visualize learned XOR prototypes
    xor_prototypes = xor_model.prototypes.data
    xor_labels = [f"XOR_Class_{i}" for i in range(xor_prototypes.shape[0])]

    # Function creates its own figure - no need for plt.figure()
    plot_prototype_space(
        prototypes=xor_prototypes,
        prototype_labels=xor_labels,
        title="XOR Problem: Learned Prototype Space",
    )
    plt.show()

    print("🎯 XOR prototypes show how TNNs learn logical relationships!")

except Exception as e:
    print(
        f"   Note: XOR prototype visualization not available ({type(e).__name__}: {e})"
    )

print("\n🎉 Visualization demo completed!")
print("   Try modifying the parameters to explore different visualizations.")
