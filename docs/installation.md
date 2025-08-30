# Installation

## Requirements

- Python 3.8 or higher
- PyTorch 1.10.0 or higher
- NumPy 1.19.0 or higher

## Install from Source (Recommended)

Currently, Verskyt is best installed from source for development and early access:

```bash
git clone https://github.com/verskyt/verskyt.git
cd verskyt
pip install -e ".[dev]"
```

This installs the package in development mode with all dependencies needed for development, testing, and documentation generation.

## Install for Basic Usage

If you only need the core functionality:

```bash
git clone https://github.com/verskyt/verskyt.git
cd verskyt  
pip install -e .
```

## Optional Dependencies

### Visualization
For enhanced plotting and visualization capabilities:
```bash
pip install -e ".[visualization]"
```

### Benchmarks  
For running paper benchmarks and comparisons:
```bash
pip install -e ".[benchmarks]"
```

### All Dependencies
To install everything:
```bash
pip install -e ".[dev,visualization,benchmarks]"
```

## Verify Installation

Test that the installation worked:

```python
import torch
from verskyt import TverskyProjectionLayer

# Create a simple layer
layer = TverskyProjectionLayer(
    in_features=4,
    num_prototypes=2, 
    num_features=8
)

# Test forward pass
x = torch.randn(3, 4)
output = layer(x)
print(f"Success! Output shape: {output.shape}")
```

You should see: `Success! Output shape: torch.Size([3, 2])`

## Development Setup

For contributors and researchers who want to run tests and contribute:

```bash
# Clone and install
git clone https://github.com/verskyt/verskyt.git
cd verskyt
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run tests to verify everything works
pytest

# Check code quality
pre-commit run --all-files
```

## Troubleshooting

### PyTorch Installation Issues
If you encounter PyTorch installation issues, install PyTorch first following the [official instructions](https://pytorch.org/get-started/locally/), then install Verskyt.

### Import Errors
If you get import errors, ensure your Python environment has access to the installed package:
```python
import sys
print(sys.path)  # Should include the path to verskyt
```

### Development Dependencies
If development commands fail, ensure you installed with dev dependencies:
```bash
pip install -e ".[dev]"
```