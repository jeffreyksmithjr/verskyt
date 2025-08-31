# Verskyt: Tversky Neural Networks

```{include} ../README.md
:start-after: # Verskyt
:end-before: ## Contributing
```

## Documentation Contents

```{toctree}
:maxdepth: 2
:caption: User Guide

installation
quickstart
tutorials/index
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
```

```{toctree}
:maxdepth: 2
:caption: Development

implementation/index
requirements/index
DEVELOPMENT
```

```{toctree}
:maxdepth: 1
:caption: Project

PROJECT_STATUS
```

## Key Features

- **🧠 Psychologically-Plausible Similarity**: Based on Tversky's feature-based similarity theory
- **🔥 Non-linear Capability**: Single layer can solve XOR (impossible for linear layers)
- **📈 Performance Gains**: Up to 24.7% accuracy improvement on complex datasets
- **⚡ Parameter Efficiency**: Fewer parameters with better performance
- **🔍 Interpretability**: Learned prototypes and features are human-recognizable
- **🔌 Drop-in Compatibility**: Easy replacement for `nn.Linear` layers

## Quick Links

- **[GitHub Repository](https://github.com/verskyt/verskyt)**
- **[Paper: "Tversky Neural Networks"](https://arxiv.org/abs/2506.11035)**
- **[Installation Guide](installation.md)**
- **[Quick Start Tutorial](quickstart.md)**

## Citation

If you use Verskyt in your research, please cite:

```bibtex
@article{doumbouya2025tversky,
  title={Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity},
  author={Doumbouya, Moussa Koulako Bala and Jurafsky, Dan and Manning, Christopher D.},
  journal={arXiv preprint arXiv:2506.11035},
  year={2025}
}
```
