# API Reference

Complete API documentation for all Verskyt components.

## Core Modules

### [verskyt.core.similarity](core.md) *(Coming Soon)*
**Mathematical foundation** - Core Tversky similarity computations

**Key functions:**
- `tversky_similarity()` - Main similarity computation
- `compute_feature_membership()` - Feature membership scoring
- `compute_salience()` - Salience weighting calculations
- `intersection_measures()` - Various intersection reduction methods
- `difference_measures()` - Various difference reduction methods

**Enums:**
- `IntersectionMethod` - Available intersection reduction methods
- `DifferenceMethod` - Available difference reduction methods

### [verskyt.layers](layers.md) *(Coming Soon)*
**Neural network components** - PyTorch-compatible layer implementations

**Classes:**
- `TverskySimilarityLayer` - Computes similarity between two inputs
- `TverskyProjectionLayer` - Projects inputs to similarity space
- `TverskyLinear` - Drop-in replacement for nn.Linear
- `TverskyAttention` - Attention mechanism using Tversky similarity

### [verskyt.utils](utils.md) *(Coming Soon)*
**Utility functions** - Helper functions and initialization strategies

**Modules:**
- `initializers` - Custom weight initialization strategies
- `metrics` - Evaluation metrics and validation functions
- `visualization` - Parameter and result visualization tools

## Usage Patterns

### Basic Similarity Computation
```python
from verskyt.core.similarity import tversky_similarity
from verskyt.core.similarity import IntersectionMethod, DifferenceMethod

# Compute similarity between two objects
sim = tversky_similarity(
    a, b, features,
    alpha=0.5, beta=0.5, theta=1.0,
    intersection_method=IntersectionMethod.PRODUCT,
    difference_method=DifferenceMethod.IGNORE_MATCH
)
```

### Layer Integration
```python
from verskyt.layers import TverskyProjectionLayer
import torch.nn as nn

class TverskyClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_features=64):
        super().__init__()
        self.backbone = nn.Sequential(...)
        self.classifier = TverskyProjectionLayer(
            in_features=input_dim,
            num_prototypes=num_classes,
            num_features=num_features
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
```

### Parameter Sharing
```python
from verskyt.layers import TverskyProjectionLayer

# Shared feature bank across layers
shared_features = torch.randn(64, 128)

layer1 = TverskyProjectionLayer(..., shared_features=shared_features)
layer2 = TverskyProjectionLayer(..., shared_features=shared_features)
```

## Parameter Specifications

### Shape Conventions
- **Input objects**: `[batch_size, in_features]`
- **Prototypes**: `[num_prototypes, in_features]`
- **Features**: `[num_features, in_features]`
- **Output similarities**: `[batch_size, num_prototypes]`

### Default Values
- **alpha**: 0.5 (distinctive features of object a)
- **beta**: 0.5 (distinctive features of object b)
- **theta**: 1.0 (common features weight)
- **intersection_method**: 'product'
- **difference_method**: 'ignorematch'

### Initialization Strategies
- **Prototypes**: Uniform distribution (-1, 1) recommended
- **Features**: Uniform distribution (-1, 1) recommended
- **Alpha, Beta, Theta**: Small positive values (0.1-1.0)

## Error Handling

### Common Issues
- **Shape mismatches**: Ensure compatible tensor shapes
- **Numerical instability**: Use appropriate epsilon values
- **Gradient vanishing**: Check parameter initialization
- **Convergence failure**: Try different initialization or hyperparameters

### Debug Functions
```python
from verskyt.utils.debug import check_gradients, validate_similarity

# Check gradient flow
check_gradients(model, sample_input)

# Validate similarity properties
validate_similarity(similarity_fn, test_cases)
```

## Performance Considerations

### Memory Usage
- Feature banks increase parameter count
- Batch processing is more memory efficient
- Consider feature sharing for large models

### Computational Complexity
- O(batch_size × num_prototypes × num_features) per layer
- Intersection/difference computations are parallelizable
- GPU acceleration recommended for large feature banks

### Optimization Tips
- Use `torch.jit.script` for production deployments
- Batch similar operations when possible
- Profile memory usage with large feature banks

## Version Compatibility

### PyTorch Versions
- **Minimum**: PyTorch 1.9.0
- **Recommended**: PyTorch 2.0+
- **GPU Support**: CUDA 11.0+

### Python Versions
- **Minimum**: Python 3.8
- **Recommended**: Python 3.9+
- **Type Hints**: Full support for static analysis
