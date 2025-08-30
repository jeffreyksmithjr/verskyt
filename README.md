# Verskyt

[![CI](https://github.com/jeffreyksmithjr/verskyt/workflows/CI/badge.svg)](https://github.com/jeffreyksmithjr/verskyt/actions/workflows/ci.yml)
[![Pre-commit](https://github.com/jeffreyksmithjr/verskyt/workflows/Pre-commit/badge.svg)](https://github.com/jeffreyksmithjr/verskyt/actions/workflows/pre-commit.yml)
[![codecov](https://codecov.io/gh/jeffreyksmithjr/verskyt/branch/main/graph/badge.svg)](https://codecov.io/gh/jeffreyksmithjr/verskyt)

An independent, research-focused Python library implementing Tversky Neural Networks (TNNs) with emphasis on **modularity**, **introspection**, and **extensibility**. Based on the psychologically plausible deep learning approach described in "Tversky Neural Networks" (Doumbouya et al., 2025).

> **Note**: This is not the official implementation by the paper authors, but rather an independent library designed to make Tversky similarity concepts accessible and extensible for researchers and practitioners.

## Why Verskyt?

Tversky Neural Networks represent a breakthrough in psychologically-motivated deep learning, offering non-linear capabilities that surpass traditional linear layers while maintaining interpretability. Verskyt makes these concepts accessible through a modular, research-friendly implementation.

**🔬 Research-First Design:**
- **Modular Architecture**: Clean separation between similarity computation, neural layers, and utilities
- **Deep Introspection**: Access and modify learned prototypes, features, and similarity parameters
- **Extensible Framework**: Easy to experiment with new similarity measures and reduction methods
- **Reproducible Science**: Comprehensive benchmarks validating paper results

**🚀 Practical Benefits:**
- **Non-linear Single Layers**: Solve XOR and complex patterns with one layer (impossible for linear layers)
- **Drop-in Compatibility**: Replace `nn.Linear` with `TverskyProjectionLayer` in existing models
- **Interpretable Representations**: Human-recognizable learned prototypes and features
- **Performance Gains**: Demonstrated improvements on vision and NLP tasks

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/verskyt.git
cd verskyt

# Install for development and research
pip install -e ".[dev]"

# Verify installation
python -c "from verskyt import TverskyProjectionLayer; print('✅ Ready for research!')"
```

### Drop-in Replacement for Linear Layers

```python
import torch
import torch.nn as nn
from verskyt import TverskyProjectionLayer

# Instead of: nn.Linear(128, 10)
layer = TverskyProjectionLayer(
    in_features=128,
    num_prototypes=10,    # equivalent to output classes
    num_features=256,     # internal feature space size
    learnable_ab=True     # learn asymmetry parameters
)

# Works exactly like nn.Linear
x = torch.randn(32, 128)
output = layer(x)  # shape: [32, 10]
```

### Research-Focused: Introspection & Modification

```python
# Access learned representations (introspection)
prototypes = layer.prototypes.detach()          # what the model recognizes
features = layer.feature_bank.detach()         # basis for similarity
alpha, beta = layer.alpha.item(), layer.beta.item()  # asymmetry params

print(f"Learned {len(prototypes)} prototypes in {len(features)}-dim feature space")
print(f"Asymmetry: α={alpha:.3f} (input focus), β={beta:.3f} (prototype focus)")

# Intervention studies (extensibility)
layer.set_prototype(0, torch.zeros_like(prototypes[0]))  # zero out class 0
modified_output = layer(x)  # see how predictions change

# Custom similarity experiments
from verskyt.core import tversky_similarity
custom_sim = tversky_similarity(
    x, prototypes, features,
    alpha=0.8, beta=0.2,  # highly asymmetric
    intersection_reduction="max",  # try different aggregations
    difference_reduction="ignorematch"
)
```

## Research Capabilities & Validation

### 🔬 Modular Experimentation
- **Similarity Variants**: 6 intersection methods × 2 difference methods = 12 combinations to explore
- **Parameter Studies**: Learnable vs. fixed α, β asymmetry parameters
- **Architecture Flexibility**: Drop-in replacement for linear layers in any PyTorch model
- **Intervention Analysis**: Modify prototypes and observe behavioral changes

### 🧪 Validated Benchmarks
This implementation includes comprehensive validation against the paper's key findings:
- **✅ XOR Solvability**: Verified single-layer non-linear capability (impossible for linear layers)
- **✅ Convergence Analysis**: 11,664 configuration parameter sweep reproducing paper results
- **✅ Mathematical Correctness**: All similarity computations validated against paper equations

### 🎯 Potential Applications
Based on capabilities demonstrated in "Tversky Neural Networks" (Doumbouya et al., 2025):
- **Vision Tasks**: ResNet architectures with Tversky final layers
- **NLP Models**: Attention mechanisms using similarity-based projections
- **Few-shot Learning**: Prototype-based classification with interpretable features
- **Causal Analysis**: Intervention studies on learned representations

## Documentation

📚 **[Complete Documentation](docs/)** - Comprehensive guides and API reference

### Quick Links
- **[API Reference](docs/api/)** - Complete function and class documentation
- **[Mathematical Specifications](docs/requirements/tnn-specification.md)** - Paper equations and implementations
- **[Development Guide](docs/implementation/plan.md)** - Testing strategy and contribution guidelines
- **[Benchmarks](docs/research/)** - Validation studies and reproduction results

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
