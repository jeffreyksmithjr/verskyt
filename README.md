# Verskyt

[![CI](https://github.com/jeffreyksmithjr/verskyt/workflows/CI/badge.svg)](https://github.com/jeffreyksmithjr/verskyt/actions/workflows/ci.yml)
[![Pre-commit](https://github.com/jeffreyksmithjr/verskyt/workflows/Pre-commit/badge.svg)](https://github.com/jeffreyksmithjr/verskyt/actions/workflows/pre-commit.yml)
[![codecov](https://codecov.io/gh/jeffreyksmithjr/verskyt/branch/main/graph/badge.svg)](https://codecov.io/gh/jeffreyksmithjr/verskyt)

A comprehensive Python library implementing Tversky Neural Networks (TNNs) - psychologically plausible deep learning models based on differentiable Tversky similarity.

## Overview

Verskyt provides PyTorch-compatible implementations of Tversky similarity functions and neural network layers that can serve as drop-in replacements for traditional linear layers, offering improved interpretability and performance in many scenarios.

**Key Features:**
- **Differentiable Tversky similarity** with multiple aggregation methods  
- **Neural network layers** compatible with existing PyTorch architectures
- **Research tools** for interpretability and intervention studies
- **Comprehensive testing** with paper result reproduction

## Quick Start

```bash
# Install from source (recommended for development)
pip install -e ".[dev]"

# Basic usage
from verskyt.layers import TverskyProjectionLayer

layer = TverskyProjectionLayer(
    in_features=128,
    num_prototypes=10,
    num_features=64
)
```

## Documentation

📚 **[Complete Documentation](docs/)** - Comprehensive guides and API reference

### Quick Links
- **[Implementation Requirements](docs/requirements/tnn-specification.md)** - Complete mathematical specifications
- **[Implementation Plan](docs/implementation/plan.md)** - Development roadmap and testing strategy  
- **[API Reference](docs/api/)** - Detailed API documentation
- **[Tutorials](docs/tutorials/)** - Step-by-step usage guides
- **[Research Tools](docs/research/)** - Experimental and analysis capabilities

## Performance Highlights

Based on "Tversky Neural Networks" (Doumbouya et al., 2025):

- **🔥 Non-linear capability**: Single layer can solve XOR (impossible for linear layers)
- **📈 ResNet-50 improvement**: Up to 24.7% accuracy gain on NABirds dataset  
- **⚡ Parameter efficiency**: 34.8% fewer parameters in GPT-2 with 7.5% perplexity reduction
- **🔍 Interpretability**: Learned prototypes and features are human-recognizable

## Contributing

We welcome contributions! Please see our [development setup](docs/implementation/) and [testing requirements](docs/implementation/plan.md#testing-strategy).

### Development Setup
```bash
git clone https://github.com/your-org/verskyt.git
cd verskyt
pip install -e ".[dev]"
pytest  # Run tests
```

## Citation

```bibtex
@article{doumbouya2025tversky,
  title={Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity},
  author={Doumbouya, Moussa Koulako Bala and Jurafsky, Dan and Manning, Christopher D.},
  journal={arXiv preprint arXiv:2506.11035},
  year={2025}
}
```

## License

[License details to be added]
