Verskyt: Tversky Neural Networks
==================================

A comprehensive Python library implementing Tversky Neural Networks (TNNs) - 
psychologically plausible deep learning models based on differentiable Tversky similarity.

**Key Features:**

* 🧠 **Psychologically-Plausible Similarity**: Based on Tversky's feature-based similarity theory
* 🔥 **Non-linear Capability**: Single layer can solve XOR (impossible for linear layers)  
* 📈 **Performance Gains**: Up to 24.7% accuracy improvement on complex datasets
* ⚡ **Parameter Efficiency**: Fewer parameters with better performance
* 🔍 **Interpretability**: Learned prototypes and features are human-recognizable
* 🔌 **Drop-in Compatibility**: Easy replacement for ``nn.Linear`` layers

Quick Start
===========

.. code-block:: python

   from verskyt import TverskyProjectionLayer
   import torch

   # Create a layer (replaces nn.Linear(128, 10))
   layer = TverskyProjectionLayer(
       in_features=128,
       num_prototypes=10,
       num_features=256
   )

   # Forward pass
   x = torch.randn(32, 128)
   output = layer(x)  # shape: [32, 10]

Installation
============

.. code-block:: bash

   git clone https://github.com/verskyt/verskyt.git
   cd verskyt
   pip install -e ".[dev]"

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
   PROJECT_STATUS
   FUTURE_WORK

Citation
========

.. code-block:: bibtex

   @article{doumbouya2025tversky,
     title={Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity},
     author={Doumbouya, Moussa Koulako Bala and Jurafsky, Dan and Manning, Christopher D.},
     journal={arXiv preprint arXiv:2506.11035},
     year={2025}
   }

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`