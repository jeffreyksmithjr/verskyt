"""
Verskyt: A library for Tversky Neural Networks.

Psychologically plausible deep learning with differentiable Tversky similarity.
"""

__version__ = "0.1.0"

from verskyt.core import tversky_similarity
from verskyt.layers import TverskyProjectionLayer, TverskySimilarityLayer
from verskyt.benchmarks import (
    run_fast_xor_benchmark,
    run_full_xor_replication,
    XORBenchmark,
)

__all__ = [
    "tversky_similarity",
    "TverskyProjectionLayer",
    "TverskySimilarityLayer",
    "run_fast_xor_benchmark",
    "run_full_xor_replication",
    "XORBenchmark",
]
