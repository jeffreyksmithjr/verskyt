"""
Verskyt: A library for Tversky Neural Networks.

Psychologically plausible deep learning with differentiable Tversky similarity.
"""

__version__ = "0.2.0"

from verskyt.benchmarks import (
    XORBenchmark,
    run_fast_xor_benchmark,
    run_full_xor_replication,
)
from verskyt.core import tversky_similarity
from verskyt.interventions import (
    ConceptGrounding,
    ConceptLibrary,
    CounterfactualAnalyzer,
    FeatureGrounder,
    ImpactAssessment,
    InterventionManager,
)
from verskyt.layers import TverskyProjectionLayer, TverskySimilarityLayer

# Optional visualization imports
try:
    from verskyt.visualizations import (  # noqa: F401
        plot_prototype_space,
        visualize_prototypes_as_data,
    )

    _visualization_available = True
except ImportError:
    _visualization_available = False

__all__ = [
    "tversky_similarity",
    "TverskyProjectionLayer",
    "TverskySimilarityLayer",
    "run_fast_xor_benchmark",
    "run_full_xor_replication",
    "XORBenchmark",
    "InterventionManager",
    "CounterfactualAnalyzer",
    "ImpactAssessment",
    "FeatureGrounder",
    "ConceptGrounding",
    "ConceptLibrary",
]

# Add visualization functions to __all__ if available
if _visualization_available:
    __all__.extend(
        [
            "plot_prototype_space",
            "visualize_prototypes_as_data",
        ]
    )
