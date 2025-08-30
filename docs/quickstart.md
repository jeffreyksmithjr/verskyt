# Quick Start Guide

Get up and running with Verskyt in minutes! This guide shows you how to use Tversky Neural Network layers as drop-in replacements for standard PyTorch layers.

## Basic Usage

### TverskyProjectionLayer as nn.Linear Replacement

The most common use case is replacing `nn.Linear` with `TverskyProjectionLayer`:

```python
import torch
import torch.nn as nn
from verskyt import TverskyProjectionLayer

# Instead of: nn.Linear(128, 10)
layer = TverskyProjectionLayer(
    in_features=128,
    num_prototypes=10,    # equivalent to out_features
    num_features=64       # internal feature space size
)

# Works exactly like nn.Linear
x = torch.randn(32, 128)  # batch of 32 samples
output = layer(x)         # shape: [32, 10]
```

### Simple Classification Model

Here's a complete example replacing linear layers in a classifier:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from verskyt import TverskyProjectionLayer

class TverskyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        
        # Replace linear layers with Tversky layers
        self.layer1 = TverskyProjectionLayer(
            in_features=input_dim,
            num_prototypes=hidden_dim,
            num_features=hidden_dim * 2,  # larger feature space
            learnable_ab=True
        )
        
        self.layer2 = TverskyProjectionLayer(
            in_features=hidden_dim,
            num_prototypes=num_classes,
            num_features=hidden_dim,
            learnable_ab=True
        )
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)  # logits for classification
        return x

# Create model
model = TverskyNet(input_dim=784, hidden_dim=256, num_classes=10)

# Use like any PyTorch model
x = torch.randn(64, 784)  # MNIST-like input
logits = model(x)         # shape: [64, 10]
```

### XOR Problem (Showcasing Non-linearity)

Tversky layers can solve XOR with a single layer, unlike linear layers:

```python
import torch
import torch.optim as optim
from verskyt import TverskyProjectionLayer

# XOR data
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([0., 1., 1., 0.])

# Single Tversky layer (impossible with single linear layer!)
model = TverskyProjectionLayer(
    in_features=2,
    num_prototypes=1,   # single output
    num_features=4,     # feature space
    learnable_ab=True
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X).squeeze()
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Test the learned function
with torch.no_grad():
    predictions = model(X).squeeze()
    print("XOR Results:")
    for i in range(4):
        print(f"Input: {X[i].numpy()}, Target: {y[i]:.0f}, Predicted: {predictions[i]:.3f}")
```

## Key Parameters

### TverskyProjectionLayer Parameters

- **`in_features`**: Input dimension (like `nn.Linear`)
- **`num_prototypes`**: Output dimension (like `out_features` in `nn.Linear`)
- **`num_features`**: Size of internal feature space (key hyperparameter)
- **`alpha, beta`**: Tversky asymmetry parameters (0.5, 0.5 = symmetric)
- **`learnable_ab`**: Whether α, β are trainable (usually `True`)

### Choosing `num_features`

The `num_features` parameter controls the expressiveness:
- **Small** (≈ `in_features`): More constrained, faster
- **Large** (≈ 2-4× `in_features`): More expressive, slower
- **Rule of thumb**: Start with 2× `in_features`

## Advanced Usage

### Shared Feature Banks

Share feature representations across layers:

```python
# Create shared feature bank
shared_features = nn.Parameter(torch.randn(128, 64))

layer1 = TverskyProjectionLayer(64, 32, 128, shared_feature_bank=shared_features)
layer2 = TverskyProjectionLayer(32, 10, 128, shared_feature_bank=shared_features)
```

### Custom Similarity Computation

Use lower-level similarity functions:

```python
from verskyt.core import tversky_similarity

# Direct similarity computation
x = torch.randn(8, 16)          # inputs
prototypes = torch.randn(5, 16)  # prototypes
features = torch.randn(32, 16)   # feature bank

similarities = tversky_similarity(
    x, prototypes, features,
    alpha=0.7, beta=0.3,          # asymmetric
    intersection_reduction="product",
    difference_reduction="substractmatch"
)
print(similarities.shape)  # [8, 5]
```

### Interpretability

Access learned prototypes and features:

```python
layer = TverskyProjectionLayer(10, 5, 20)

# Get learned prototypes (what the model recognizes)
prototypes = layer.prototypes.detach()
print(f"Prototype 0: {prototypes[0]}")

# Get learned features (basis for similarity computation)  
features = layer.feature_bank.detach()
print(f"Feature 0: {features[0]}")

# Modify prototypes for intervention studies
layer.set_prototype(0, torch.zeros(10))  # zero out prototype 0
```

## Next Steps

- **[Full API Reference](api/index.md)**: Complete documentation of all functions and classes
- **[Tutorials](tutorials/index.md)**: In-depth guides and examples
- **[Paper](https://arxiv.org/abs/2506.11035)**: Mathematical foundations and experimental results

## Common Patterns

### Replace Linear Layers
```python
# Before
self.classifier = nn.Linear(512, 10)

# After  
self.classifier = TverskyProjectionLayer(512, 10, 1024)
```

### Add to Existing Models
```python
# ResNet-style with Tversky final layer
class ResNetWithTversky(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        # Replace final layer
        self.backbone.fc = TverskyProjectionLayer(512, 1000, 1024)
```