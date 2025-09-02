Verskyt: Tversky Neural Networks
==================================


.. image:: https://zenodo.org/badge/1047467589.svg
   :target: https://doi.org/10.5281/zenodo.17014431

Verskyt (pronounced "ver-SKIT") is a Python library for Tversky Neural Networks (TNNs) built on three design principles: **Modularity**, **Introspection**, and **Extensibility**. Verskyt provides PyTorch-compatible TNN implementations alongside tools for model introspection and prototype analysis.

TNNs are psychologically plausible deep learning models based on differentiable Tversky similarity that operate by projecting inputs into a learned feature space (Ω), where similarity to explicit prototypes (Π) is computed.

**Design Principles:**

* 🔧 **Modularity**: Clean, composable components that integrate with PyTorch
* 🔍 **Introspection**: Tools for examining model internals and decision processes
* 🚀 **Extensibility**: Built for researchers to modify and develop TNN architectures

**Key Features:**

* 🧠 **Psychologically-Plausible Similarity**: Based on Tversky's feature-based similarity theory
* 🔥 **Non-linear Capability**: Single layer can solve XOR (impossible for linear layers)
* 📈 **Performance Gains**: Up to 24.7% accuracy improvement on complex datasets
* ⚡ **Parameter Efficiency**: Fewer parameters with better performance
* 🔌 **Drop-in Compatibility**: Easy replacement for ``nn.Linear`` layers

Quick Start
===========

.. code-block:: python

   from verskyt import TverskyProjectionLayer
   import torch

   # Create a layer (replaces nn.Linear(128, 10))
   layer = TverskyProjectionLayer(
       in_features=128,      # Dimensionality of the input vector
       num_prototypes=10,    # Corresponds to output classes
       num_features=256      # Dimensionality of the internal learned feature space (Ω)
   )

   # Forward pass
   x = torch.randn(32, 128)
   output = layer(x)  # shape: [32, 10]

Installation
============

.. code-block:: bash

   pip install verskyt

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   DEVELOPMENT
   DEPLOYMENT
   PROJECT_STATUS
   FUTURE_WORK

Citation
========

If you use Verskyt in your research, please cite both the original Tversky Neural Network paper and this library.

**1. Foundational Paper:**

.. code-block:: bibtex

   @article{doumbouya2025tversky,
     title={Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity},
     author={Doumbouya, Moussa Koulako Bala and Jurafsky, Dan and Manning, Christopher D.},
     journal={arXiv preprint arXiv:2506.11035},
     year={2025}
   }

**2. This Library (Verskyt):**

We recommend citing the specific version of the software you used. You can get a persistent DOI for each version from `Zenodo <https://zenodo.org>`_.

.. code-block:: bibtex

   @software{smith_2025_verskyt,
     author       = {Smith, Jeff},
     title        = {{Verskyt: A versatile toolkyt for Tversky Neural Networks}},
     month        = aug,
     year         = 2025,
     publisher    = {Zenodo},
     version      = {v0.2.2},
     doi          = {10.5281/zenodo.17014431},
     url          = {https://doi.org/10.5281/zenodo.17014431}
   }


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
