# Tutorials

Step-by-step guides for using Verskyt in various scenarios.

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- Basic understanding of neural networks

### Installation
```bash
pip install verskyt
# or for development
pip install -e ".[dev]"
```

## Tutorial Series

### [Getting Started](getting-started.md) *(Coming Soon)*
Quick setup and your first Tversky model
- Installation and environment setup
- Basic TverskyProjectionLayer usage
- Simple classification example

### [Basic Usage](basic-usage.md) *(Coming Soon)*
Core functionality and common patterns
- Understanding Tversky similarity
- Building models with Tversky layers
- Training and evaluation basics

### [Advanced Usage](advanced-usage.md) *(Coming Soon)*
Complex scenarios and optimizations
- Parameter sharing strategies
- Integration with existing models (ResNet, GPT-2)
- Performance optimization techniques
- Custom similarity functions

## Example Scenarios

### Computer Vision
- **Image Classification**: Using TverskyProjectionLayer as classifier head
- **Feature Learning**: Interpretable feature discovery
- **Transfer Learning**: Adapting pre-trained models with Tversky layers

### Natural Language Processing
- **Language Modeling**: Replacing linear layers in transformers
- **Text Classification**: Similarity-based text understanding
- **Attention Mechanisms**: Asymmetric attention with Tversky similarity

### Research Applications
- **Interpretability**: Understanding learned representations
- **Intervention Studies**: Manipulating prototypes and features
- **Psychological Modeling**: Studying human-like similarity

## Code Examples

Quick examples to get you started:

```python
import torch
from verskyt.layers import TverskyProjectionLayer

# Simple classification layer
layer = TverskyProjectionLayer(
    in_features=128,
    num_prototypes=10,  # 10 classes
    num_features=64     # 64 learned features
)

# Forward pass
x = torch.randn(32, 128)  # batch of 32 samples
similarities = layer(x)   # [32, 10] similarities to prototypes
```

```python
from verskyt.core.similarity import tversky_similarity

# Direct similarity computation
a = torch.randn(128)
b = torch.randn(128)
features = torch.randn(64, 128)

sim = tversky_similarity(
    a, b, features,
    alpha=0.5, beta=0.5, theta=1.0,
    intersection_method='product',
    difference_method='ignorematch'
)
```

## Best Practices

### Model Design
- Start with proven architectures (ResNet + TverskyHead)
- Use parameter sharing for efficiency
- Monitor convergence carefully with Tversky layers

### Training
- Use uniform initialization for prototypes and features
- Avoid normalizing vectors during similarity computation
- Consider multiple random restarts for better convergence

### Evaluation
- Always test XOR capability for single layers
- Compare against linear baselines
- Evaluate interpretability of learned prototypes

## Community Resources

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Usage questions and best practices
- **Examples Repository**: Real-world usage examples
- **Paper Reproductions**: Exact implementations of paper experiments

## Contributing Tutorials

We welcome tutorial contributions! Please see our [contribution guidelines](../../README.md) for:
- Tutorial format and style
- Code example standards
- Review process
