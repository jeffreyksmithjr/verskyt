# Verskyt

[![CI](https://github.com/jeffreyksmithjr/verskyt/workflows/CI/badge.svg)](https://github.com/jeffreyksmithjr/verskyt/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/jeffreyksmithjr/verskyt/branch/main/graph/badge.svg)](https://codecov.io/gh/jeffreyksmithjr/verskyt) [![PyPI version](https://badge.fury.io/py/verskyt.svg)](https://badge.fury.io/py/verskyt)

`verskyt` is an independent, research-focused Python library for Tversky Neural Networks (TNNs). It provides a modular, introspective, and extensible implementation of the psychologically plausible deep learning models described in "Tversky Neural Networks" (Doumbouya et al., 2025).

This library is designed to be a foundational tool for researchers exploring novel neuro-symbolic architectures, interpretable representations, and causal analysis.

## Why Verskyt?

Tversky Neural Networks offer a new paradigm for building interpretable models by replacing standard linear projections with a similarity-based mechanism grounded in cognitive science. `verskyt` provides the tools to both leverage and extend these capabilities.

#### 🧠 A Faithful & Extensible TNN Implementation

  * **Drop-in Compatibility**: Replace `torch.nn.Linear` layers with `verskyt.TverskyProjectionLayer` to introduce TNN capabilities into existing PyTorch models.
  * **Full Parameter Control**: All aspects of the Tversky contrast model—prototypes (Π), features (Ω), and asymmetry parameters (α, β)—are learnable and accessible.
  * **Modular Similarity Functions**: Easily experiment with different mathematical formulations for feature intersection and difference to explore their impact on model behavior.

#### 🔬 Built for Advanced Research & Introspection

Beyond a simple implementation, `verskyt` includes a powerful toolkit for interrogating and manipulating trained models.

  * **Deep Introspection**: Programmatically access and analyze the learned prototypes and feature banks to understand what a model has learned.
  * **Causal Intervention**: Use the `InterventionManager` to perform "prototype surgery"—directly editing a model's internal concepts and simulating counterfactuals to causally probe its logic.

## Quick Start

Install from PyPI:
`pip install verskyt`

### Basic Usage: Drop-in Replacement

`verskyt` layers are designed to be a seamless replacement for standard PyTorch layers.

```python
import torch
from verskyt.layers import TverskyProjectionLayer

# A TNN layer that can replace nn.Linear(in_features=128, out_features=10)
layer = TverskyProjectionLayer(
    in_features=128,
    num_prototypes=10,    # Corresponds to output classes
    num_features=256,     # Size of the internal feature space
)

# It works just like a standard PyTorch layer
x = torch.randn(32, 128)
output = layer(x)  # shape: [32, 10]
```

### Advanced Usage: Introspection & Intervention

Go beyond prediction and start interrogating your model's logic with the built-in intervention toolkit.

```python
from verskyt.interventions import InterventionManager

# Assume 'model' is a trained model with TverskyProjectionLayer
manager = InterventionManager(model)

# 1. Inspect the model's learned concepts
prototypes = manager.list_prototypes()
print(f"Inspecting {len(prototypes)} learned prototypes.")

# 2. Examine individual prototypes and features
proto_info = manager.get_prototype("layer_name", 0)
print(f"Prototype 0: shape={proto_info.shape}, norm={proto_info.norm:.3f}")

# 3. Permanently edit a prototype ("prototype surgery")
original_proto = manager.get_prototype("layer_name", 0)
modified_vector = original_proto.vector * 0.5  # Dampen the prototype
manager.modify_prototype("layer_name", 0, modified_vector)

# 4. Reset to original state when done
manager.reset_to_original()
```

## Features & Status

This library provides a comprehensive implementation of Tversky Neural Networks, validated against the original paper's specifications.

| Feature Area | Component | Status |
| :--- | :--- | :--- |
| **Core TNN Layers** | `TverskyProjectionLayer` | ✅ **Complete** |
| | `TverskySimilarityLayer` | ✅ **Complete** |
| **Similarity Functions** | Intersection Reductions | 🟡 **Partial** (3/6 implemented: `product`, `min`, `mean`) |
| | Difference Reductions | ✅ **Complete** (Both `substractmatch` & `ignorematch`) |
| **Validation** | XOR Non-Linear Benchmark | ✅ **Complete** (Convergence verified) |
| **Research Toolkit**| `InterventionManager` API | ✅ **Complete** (Inspect, Edit, Simulate) |
| | `FeatureGrounder` Framework | ✅ **Complete** |
| | Visualization Suite | ⏳ **On Roadmap** |
| **Model Zoo** | ResNet Integration | ⏳ **On Roadmap** |

## 🚀 Roadmap

`verskyt` is under active development. Key priorities for upcoming releases include:

  * [ ] **Complete Specification Compliance**: Implement the remaining intersection reduction methods (`max`, `gmean`, `softmin`) for full compliance with the original paper.
  * [ ] **Visualization Suite**: Add powerful tools for visualizing prototypes in the data domain, plotting decision boundaries, and analyzing the impact of interventions.
  * [ ] **Expanded Model Zoo**: Provide pre-built `TverskyResNet` and other architectures to benchmark performance on standard vision datasets like MNIST and CIFAR-10.
  * [ ] **Performance Optimizations**: Profile and optimize the core similarity computations for large-scale training.

## Documentation

For complete usage guides, tutorials, and the API reference, please see the **[Full Documentation Website](https://verskyt.readthedocs.io)**.

## Contributing

Contributions are welcome! Please see our development and contribution guidelines.

## Citation

To cite the foundational TNN paper:

```bibtex
@article{doumbouya2025tversky,
  title={Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity},
  author={Doumbouya, Moussa Koulako Bala and Jurafsky, Dan and Manning, Christopher D.},
  journal={arXiv preprint arXiv:2506.11035},
  year={2025}
}
```

To cite this library:
(BibTeX citation for `verskyt` to be added upon first archival release)
