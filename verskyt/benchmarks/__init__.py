"""
Benchmark utilities for Tversky Neural Networks.

This module provides benchmark suites for validating TNN implementations
against paper specifications and performance baselines.
"""

from .xor_suite import (
    XORBenchmark,
    run_fast_xor_benchmark,
    run_full_xor_replication,
    XORConfig,
    FAST_BENCHMARK_CONFIG,
    FULL_PAPER_CONFIG,
)

__all__ = [
    "XORBenchmark",
    "run_fast_xor_benchmark", 
    "run_full_xor_replication",
    "XORConfig",
    "FAST_BENCHMARK_CONFIG",
    "FULL_PAPER_CONFIG",
]