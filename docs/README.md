# Verskyt Documentation

A complete implementation of Tversky Neural Networks (TNNs) with full PyTorch integration.

## Documentation Overview

### 📋 [Requirements](requirements/)
Complete specifications and analysis of TNNs based on the original paper.

- **[TNN Specification](requirements/tnn-specification.md)** - Complete implementation requirements
- **[Paper Analysis](requirements/)** - Analysis of core concepts and mathematical formulations

### 🏗️ [Implementation](implementation/)
Detailed implementation guides and architectural decisions.

- **[Implementation Plan](implementation/plan.md)** - Phased development approach with validation criteria
- **[Architecture Overview](implementation/)** - System design and component relationships
- **[Testing Strategy](implementation/)** - Complete testing approach

### 📚 [Tutorials](tutorials/)
Step-by-step guides for using Verskyt.

- **[Getting Started](tutorials/)** - Quick setup and basic usage
- **[Basic Usage](tutorials/)** - Core functionality examples
- **[Advanced Usage](tutorials/)** - Complex scenarios and optimizations

### 🔧 [API Reference](api/)
Complete API documentation for all components.

- **[Core Module](api/)** - Similarity functions and mathematical operations
- **[Layers Module](api/)** - Neural network layer implementations
- **[Utils Module](api/)** - Utility functions and helpers

### 🔬 [Research](research/)
Research tools and experimental features.

- **[Experiments](research/)** - Reproducible experimental setups
- **[Interpretability](research/)** - Visualization and analysis tools

## Quick Navigation

### For Researchers
- Start with [TNN Specification](requirements/tnn-specification.md) for complete mathematical formulations
- See [Experiments](research/) for reproducing paper results

### For Developers
- Begin with [Implementation Plan](implementation/plan.md) for development roadmap
- Check [API Reference](api/) for component details

### For Users
- Start with [Getting Started](tutorials/) for installation and basic usage
- See [Basic Usage](tutorials/) for common use cases

## Project Structure

```
verskyt/
├── verskyt/                 # Main package
│   ├── core/               # Core similarity functions
│   ├── layers/             # Neural network layers
│   ├── utils/              # Utility functions
│   └── research/           # Research tools
├── tests/                  # Test suite
├── docs/                   # Documentation (you are here)
└── examples/               # Usage examples
```

## Contributing

See the main [README](../README.md) for contribution guidelines and development setup.

## References

Based on: "Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity" (Doumbouya et al., 2025)
