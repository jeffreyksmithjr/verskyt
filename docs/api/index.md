# API Reference

Complete reference for all Verskyt functions, classes, and modules.

## Core Modules

```{toctree}
:maxdepth: 2

core
layers
utils
benchmarks
```

## Quick Reference

### Main Classes

- {class}`verskyt.TverskyProjectionLayer` - Main projection layer (drop-in for `nn.Linear`)
- {class}`verskyt.TverskySimilarityLayer` - Pairwise similarity computation layer

### Core Functions

- {func}`verskyt.core.tversky_similarity` - Core Tversky similarity computation
- {func}`verskyt.core.compute_feature_membership` - Feature membership scores
- {func}`verskyt.core.compute_salience` - Object salience computation

### Enums

- {class}`verskyt.core.IntersectionReduction` - Methods for feature intersection aggregation
- {class}`verskyt.core.DifferenceReduction` - Methods for feature difference computation

## Import Patterns

### Standard Imports
```python
from verskyt import TverskyProjectionLayer, TverskySimilarityLayer
```

### Core Functions
```python
from verskyt.core import tversky_similarity, compute_salience
```

### Advanced Usage
```python
from verskyt.core.similarity import (
    IntersectionReduction, 
    DifferenceReduction,
    tversky_contrast_similarity
)
```

### Benchmarks
```python
from verskyt.benchmarks import XORBenchmark
```