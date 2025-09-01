# verskyt.visualizations

Visualization tools for Tversky Neural Networks.

This module provides functions for visualizing and interpreting learned prototypes
and features in TNNs. The functions are designed to make abstract concepts of
"prototypes" and "features" tangible and visible for research analysis.

## Module: plotting

```{eval-rst}
.. automodule:: verskyt.visualizations.plotting
   :members:
   :undoc-members:
   :show-inheritance:
```

## Functions

### plot_prototype_space

```{eval-rst}
.. autofunction:: verskyt.visualizations.plotting.plot_prototype_space
```

### visualize_prototypes_as_data

```{eval-rst}
.. autofunction:: verskyt.visualizations.plotting.visualize_prototypes_as_data
```

## Usage Examples

### Basic Prototype Space Visualization

```python
import torch
from verskyt.visualizations import plot_prototype_space

# Assume you have trained prototypes
prototypes = model.tnn_layer.prototypes
labels = ["Low-Risk", "Medium-Risk", "High-Risk"]

# Visualize the learned prototype space
ax = plot_prototype_space(prototypes, labels)
plt.show()
```

### Data-Based Prototype Interpretation

```python
from verskyt.visualizations import visualize_prototypes_as_data

# Show which data samples are most similar to each prototype
fig = visualize_prototypes_as_data(
    encoder=model.encoder,
    prototypes=model.tnn_layer.prototypes,
    prototype_labels=["Class 0", "Class 1"],
    dataloader=train_loader,
    top_k=5
)
plt.show()
```

## Requirements

The visualization module requires additional dependencies that can be installed with:

```bash
pip install verskyt[visualization]
```

Dependencies include:

- matplotlib>=3.5.0
- seaborn>=0.12.0
- scikit-learn>=1.1.0
