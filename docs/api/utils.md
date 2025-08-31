# verskyt.utils

Utility functions and helper modules.

## Module: initializers

```{eval-rst}
.. automodule:: verskyt.utils.initializers
   :members:
   :undoc-members:
   :show-inheritance:
```

Custom initialization strategies for Tversky Neural Network parameters.

## Functions

### initialize_prototypes

```{eval-rst}
.. autofunction:: verskyt.utils.initializers.initialize_prototypes
```

### initialize_features

```{eval-rst}
.. autofunction:: verskyt.utils.initializers.initialize_features
```

### initialize_tversky_params

```{eval-rst}
.. autofunction:: verskyt.utils.initializers.initialize_tversky_params
```

## Usage Examples

### Custom Initialization

```python
from verskyt.utils.initializers import initialize_prototypes
import torch

# Initialize prototypes with custom strategy
prototypes = torch.empty(10, 64)
initialize_prototypes(prototypes, method="xavier_uniform")
```
