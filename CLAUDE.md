# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Verskyt is a Python library implementing Tversky Neural Networks (TNNs) - psychologically plausible deep learning models based on differentiable Tversky similarity. The library provides PyTorch-based implementations of similarity computations and neural network layers.

## Development Commands

### Testing
```bash
# Run all tests with coverage
pytest -v --cov=verskyt --cov-report=term-missing

# Run specific test file
pytest tests/test_basic_functionality.py -v

# Run specific test
pytest tests/test_basic_functionality.py::TestBasicSimilarity::test_similarity_shape -v
```

### Code Quality
```bash
# Format code with black
black verskyt tests

# Sort imports
isort verskyt tests

# Lint with flake8
flake8 verskyt tests

# Type checking
mypy verskyt
```

### Build and Install
```bash
# Install in development mode with all dependencies
pip install -e ".[dev]"

# Install with visualization tools
pip install -e ".[visualization]"

# Install with benchmark dependencies
pip install -e ".[benchmarks]"
```

## Architecture

The library is organized into modular components:

- **verskyt/core/similarity.py**: Core mathematical operations for Tversky similarity
  - `tversky_similarity()`: Main similarity computation function
  - `compute_feature_membership()`: Feature membership scoring
  - `compute_salience()`: Salience weighting calculations
  - Enums for intersection/difference reduction methods

- **verskyt/layers/**: Neural network layers
  - `TverskyProjectionLayer`: Projects inputs to feature space
  - `TverskySimilarityLayer`: Computes similarity between inputs and prototypes

- **verskyt/utils/**: Utility functions
  - `initializers.py`: Custom weight initialization strategies

## Key Implementation Details

- Based on paper: "Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity" (Doumbouya et al., 2025)
- All operations are differentiable and PyTorch-compatible
- Similarity values are normalized to [0, 1] range
- Supports batch processing with shape conventions:
  - Objects/inputs: [batch_size, in_features]
  - Prototypes: [num_prototypes, in_features]
  - Features: [num_features, in_features]

## Testing Conventions

- Test files are in `tests/` directory following `test_*.py` naming
- Tests are organized by functionality (core, layers, utils)
- Use pytest fixtures for common test data
- Ensure all similarity outputs are in valid [0, 1] range
- Test gradient flow for all differentiable operations