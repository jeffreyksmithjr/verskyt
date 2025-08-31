# Verskyt
*A versatile toolkyt for Tversky Neural Networks*

[![CI](https://github.com/jeffreyksmithjr/verskyt/workflows/CI/badge.svg)](https://github.com/jeffreyksmithjr/verskyt/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/jeffreyksmithjr/verskyt/branch/main/graph/badge.svg)](https://codecov.io/gh/jeffreyksmithjr/verskyt) [![PyPI version](https://badge.fury.io/py/verskyt.svg)](https://badge.fury.io/py/verskyt)

**Verskyt** is a comprehensive Python library that implements Tversky Neural Networks (TNNs) with advanced research and analysis capabilities. Beyond providing faithful PyTorch-compatible TNN layers, Verskyt offers a complete toolkit for model introspection, causal intervention, and prototype analysis—making it a foundational platform for researchers exploring interpretable deep learning.

## What are Tversky Neural Networks?

Tversky Neural Networks represent a novel paradigm in deep learning, introduced by Doumbouya et al. (2025). TNNs replace traditional linear transformations with **similarity-based computations** grounded in cognitive science, specifically Tversky's feature-based similarity theory.

**Key TNN Properties:**
- **Psychologically Plausible**: Based on established cognitive models of human similarity perception
- **Asymmetric Similarity**: Can learn that "A is more similar to B than B is to A" (unlike standard neural networks)
- **Interpretable Representations**: Uses explicit prototypes and feature sets that can be directly examined
- **Non-linear Single Layer**: Can solve non-linearly separable problems (like XOR) with just one layer

## What Verskyt Provides

While TNNs define the mathematical framework, **Verskyt delivers the implementation plus advanced research capabilities** that go far beyond basic TNN functionality:

### 🧠 Complete TNN Implementation

**Production-Ready PyTorch Integration:**
- **Drop-in Compatibility**: Replace `torch.nn.Linear` layers with `verskyt.TverskyProjectionLayer` in existing models
- **Full Parameter Control**: All TNN components—prototypes (Π), features (Ω), and asymmetry parameters (α, β)—are learnable and accessible
- **Complete Specification**: All 6 intersection reduction methods and 2 difference methods from the original paper
- **Validated Implementation**: Passes all mathematical correctness tests, including the XOR non-linearity benchmark

### 🔬 Advanced Research Toolkit

**Verskyt's unique contribution** is a comprehensive suite of analysis tools not available elsewhere:

**Model Introspection:**
- **Prototype Analysis**: Examine learned prototype vectors and their semantic meanings
- **Feature Bank Inspection**: Understand which features the model has discovered
- **Similarity Landscape Mapping**: Visualize how the model perceives relationships between concepts

**Causal Intervention Framework:**
- **Prototype Surgery**: Directly edit model concepts and observe behavioral changes
- **Counterfactual Analysis**: Simulate "what if" scenarios by modifying internal representations
- **Concept Grafting**: Transfer learned concepts between different models

**Experimental Infrastructure:**
- **Benchmark Suites**: Comprehensive testing against paper specifications
- **Reproducible Research**: Tools for systematic hyperparameter exploration and results validation

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

## Library Implementation Status

Verskyt provides a complete, production-ready implementation of TNNs with extensive research capabilities:

| Implementation Area | Component | Status |
| :--- | :--- | :--- |
| **TNN Core** | `TverskyProjectionLayer` | ✅ **Complete** - Drop-in PyTorch compatibility |
| | `TverskySimilarityLayer` | ✅ **Complete** - All similarity computations |
| | Intersection Methods | ✅ **Complete** - All 6 from paper: `product`, `min`, `max`, `mean`, `gmean`, `softmin` |
| | Difference Methods | ✅ **Complete** - Both `substractmatch` & `ignorematch` |
| **Paper Validation** | XOR Benchmark | ✅ **Complete** - Non-linearity verified |
| | Mathematical Correctness | ✅ **Complete** - All specifications validated |
| **Research Tools** | `InterventionManager` | ✅ **Complete** - Prototype surgery & analysis |
| | `FeatureGrounder` | ✅ **Complete** - Concept mapping framework |
| | Prototype Analysis | ✅ **Complete** - Introspection APIs |
| **Development** | Comprehensive Testing | ✅ **Complete** - 60+ tests, 75% coverage |
| | CI/CD Pipeline | ✅ **Complete** - Automated quality & releases |

## 🚀 Research Roadmap

Verskyt continues expanding its research toolkit capabilities:

### Next Release (v0.2.0)
  * [ ] **Interactive Visualization Suite**: Tools for prototype visualization, similarity landscapes, and intervention impact analysis
  * [ ] **Extended Benchmark Suite**: Comprehensive evaluation across more datasets and TNN configurations
  * [ ] **Performance Profiling**: Optimization for large-scale models and training efficiency

### Future Releases
  * [ ] **TverskyResNet Implementation**: Pre-built architecture demonstrating TNN integration in complex models
  * [ ] **Concept Transfer Tools**: Framework for moving learned concepts between different TNN models
  * [ ] **Uncertainty Quantification**: Tools for measuring confidence in TNN predictions and prototype assignments
  * [ ] **Multi-Modal Extensions**: Extend TNN concepts to handle different data modalities simultaneously

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
