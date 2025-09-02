# verskyt.layers

Neural network layers implementing Tversky similarity computations.

## Module: projection

```{eval-rst}
.. automodule:: verskyt.layers.projection
   :members:
   :undoc-members:
   :show-inheritance:
```

## Classes

### TverskyProjectionLayer

```{eval-rst}
.. autoclass:: verskyt.layers.projection.TverskyProjectionLayer
   :members:
   :undoc-members:
   :special-members: __init__
```

The main layer for replacing `nn.Linear` with Tversky similarity-based projections.

**Key Methods:**
- `forward(x)` - Compute similarity to all prototypes
- `get_prototype(index)` - Access individual prototype vectors
- `set_prototype(index, value)` - Modify prototype vectors for interventions
- `reset_parameters()` - Reinitialize all parameters

**Properties:**
- `weight` - Compatibility property returning prototypes (for `nn.Linear` replacement)

### TverskySimilarityLayer

```{eval-rst}
.. autoclass:: verskyt.layers.projection.TverskySimilarityLayer
   :members:
   :undoc-members:
   :special-members: __init__
```

Layer for computing element-wise similarity between pairs of objects.

**Key Methods:**
- `forward(a, b)` - Compute similarity between object pairs
- `reset_parameters()` - Reinitialize parameters

## Usage Examples

### Basic Projection Layer

```python
import torch
from verskyt.layers import TverskyProjectionLayer

# Create layer (replaces nn.Linear(128, 10))
layer = TverskyProjectionLayer(
    in_features=128,
    num_prototypes=10,
    num_features=256,
    learnable_ab=True
)

# Forward pass
x = torch.randn(32, 128)
similarities = layer(x)  # shape: [32, 10]
```

### Pairwise Similarity Layer

```python
from verskyt.layers import TverskySimilarityLayer

# Create similarity layer
sim_layer = TverskySimilarityLayer(
    in_features=64,
    num_features=128,
    learnable_ab=True
)

# Compute pairwise similarities
a = torch.randn(32, 64)
b = torch.randn(32, 64)
similarities = sim_layer(a, b)  # shape: [32]
```

### Parameter Access and Modification

```python
layer = TverskyProjectionLayer(10, 5, 20)

# Access learned representations
prototypes = layer.prototypes.detach()
features = layer.feature_bank.detach()

# Modify specific prototype (for intervention studies)
new_prototype = torch.zeros(10)
layer.set_prototype(0, new_prototype)

# Access Tversky parameters
print(f"Alpha: {layer.alpha.item()}")
print(f"Beta: {layer.beta.item()}")
```

## Complete Examples

For comprehensive usage examples in realistic scenarios, see:

- **[Visualization Demo](../../examples/visualization_demo.py)** - Complete TNN model with visualization analysis
- **[Research Tutorial](../../examples/research_tutorial.py)** - Advanced usage patterns and intervention studies
- **[Intervention Demo](../../examples/intervention_demo.py)** - Prototype manipulation and causal analysis

### XOR Problem Demonstration

The classic XOR problem demonstrates TNN's non-linear capability with a single layer:

```python
from verskyt import TverskyProjectionLayer
import torch

# XOR data: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
xor_inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
xor_targets = torch.tensor([0, 1, 1, 0]).float()

# Single TNN layer can solve XOR (impossible for nn.Linear)
xor_model = TverskyProjectionLayer(
    in_features=2,
    num_prototypes=2,
    num_features=4,
    alpha=0.5,
    beta=0.5,
)

# Train the model (simplified)
optimizer = torch.optim.Adam(xor_model.parameters(), lr=0.1)
for epoch in range(100):
    optimizer.zero_grad()
    outputs = xor_model(xor_inputs)
    predictions = torch.softmax(outputs, dim=1)[:, 1]
    loss = torch.nn.functional.binary_cross_entropy(predictions, xor_targets)
    loss.backward()
    optimizer.step()

# Learned prototypes can be visualized with verskyt.visualizations
```

## Integration Patterns

### Drop-in Replacement for nn.Linear

```python
import torch.nn as nn
from verskyt import TverskyProjectionLayer

# Original network
class OriginalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, 10)  # ← Replace this

# Enhanced with TNN
class TNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 64)
        self.classifier = TverskyProjectionLayer(  # ← TNN replacement
            in_features=64,
            num_prototypes=10,
            num_features=128
        )
```
