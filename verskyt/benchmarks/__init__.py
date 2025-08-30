"""
Benchmark utilities for Tversky Neural Networks.

This module provides benchmark suites for validating TNN implementations
against paper specifications and performance baselines.
"""

from .xor_suite import (
    FAST_BENCHMARK_CONFIG,
    FULL_PAPER_CONFIG,
    XORBenchmark,
    XORConfig,
    run_fast_xor_benchmark,
    run_full_xor_replication,
)

__all__ = [
    "XORBenchmark",
    "run_fast_xor_benchmark",
    "run_full_xor_replication",
    "XORConfig",
    "FAST_BENCHMARK_CONFIG",
    "FULL_PAPER_CONFIG",
]
