# verskyt.benchmarks

Benchmark suites for validating paper results and comparing performance.

## Module: xor_suite

```{eval-rst}
.. automodule:: verskyt.benchmarks.xor_suite
   :members:
   :undoc-members:
   :show-inheritance:
```

XOR benchmark suite for validating non-linear learning capabilities.

## Classes

### XORBenchmark

```{eval-rst}
.. autoclass:: verskyt.benchmarks.xor_suite.XORBenchmark
   :members:
   :undoc-members:
   :special-members: __init__
```

### XORConfig

```{eval-rst}
.. autoclass:: verskyt.benchmarks.xor_suite.XORConfig
   :members:
   :undoc-members:
```

## Functions

### run_xor_benchmark

```{eval-rst}
.. autofunction:: verskyt.benchmarks.xor_suite.run_xor_benchmark
```

### analyze_convergence

```{eval-rst}
.. autofunction:: verskyt.benchmarks.xor_suite.analyze_convergence
```

## Usage Examples

### Quick XOR Test

```python
from verskyt.benchmarks import XORBenchmark

# Run fast benchmark (48 configurations)
benchmark = XORBenchmark()
results = benchmark.run_fast_benchmark()

print(f"Success rate: {results['success_rate']:.2%}")
print(f"Mean convergence time: {results['mean_epochs']:.1f} epochs")
```

### Full Paper Replication

```python
# Run complete benchmark (11,664 configurations)
results = benchmark.run_full_benchmark()

# Analyze results
convergence_analysis = benchmark.analyze_convergence(results)
print(convergence_analysis)
```