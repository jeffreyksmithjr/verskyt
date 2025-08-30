verskyt.layers
==============

Neural network layers implementing Tversky similarity computations.

Module: projection
------------------

.. automodule:: verskyt.layers.projection
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

TverskyProjectionLayer
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: verskyt.layers.projection.TverskyProjectionLayer
   :members:
   :undoc-members:
   :special-members: __init__

The main layer for replacing ``nn.Linear`` with Tversky similarity-based projections.

**Key Methods:**

* ``forward(x)`` - Compute similarity to all prototypes
* ``get_prototype(index)`` - Access individual prototype vectors
* ``set_prototype(index, value)`` - Modify prototype vectors for interventions
* ``reset_parameters()`` - Reinitialize all parameters

**Properties:**

* ``weight`` - Compatibility property returning prototypes (for ``nn.Linear`` replacement)

TverskySimilarityLayer
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: verskyt.layers.projection.TverskySimilarityLayer
   :members:
   :undoc-members:
   :special-members: __init__

Layer for computing element-wise similarity between pairs of objects.

**Key Methods:**

* ``forward(a, b)`` - Compute similarity between object pairs
* ``reset_parameters()`` - Reinitialize parameters

Usage Examples
--------------

Basic Projection Layer
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from verskyt.layers import TverskyProjectionLayer

   # Create layer (replaces nn.Linear(128, 10))
   layer = TverskyProjectionLayer(
       in_features=128,
       num_prototypes=10,
       num_features=256,
       learnable_ab=True
   )

   # Forward pass
   x = torch.randn(32, 128)
   similarities = layer(x)  # shape: [32, 10]

Pairwise Similarity Layer
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from verskyt.layers import TverskySimilarityLayer

   # Create similarity layer
   sim_layer = TverskySimilarityLayer(
       in_features=64,
       num_features=128,
       learnable_ab=True
   )

   # Compute pairwise similarities
   a = torch.randn(32, 64)
   b = torch.randn(32, 64)
   similarities = sim_layer(a, b)  # shape: [32]

Parameter Access and Modification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   layer = TverskyProjectionLayer(10, 5, 20)

   # Access learned representations
   prototypes = layer.prototypes.detach()
   features = layer.feature_bank.detach()

   # Modify specific prototype (for intervention studies)
   new_prototype = torch.zeros(10)
   layer.set_prototype(0, new_prototype)

   # Access Tversky parameters
   print(f"Alpha: {layer.alpha.item()}")
   print(f"Beta: {layer.beta.item()}")
