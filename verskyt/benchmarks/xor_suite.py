"""
XOR benchmark suite for Tversky Neural Networks.

Reproduces XOR experiments from "Tversky Neural Networks: Psychologically 
Plausible Deep Learning with Differentiable Tversky Similarity" (Doumbouya et al., 2025).

Provides both fast development benchmarks and full paper replication capabilities.
"""

import time
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Tuple, Union, Optional
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from verskyt.core.similarity import IntersectionReduction, DifferenceReduction
from verskyt.layers.projection import TverskyProjectionLayer


@dataclass
class XORConfig:
    """Configuration for XOR benchmark experiments."""
    
    intersection_methods: List[Union[str, IntersectionReduction]] = field(
        default_factory=lambda: ["product", "mean", "max", "gmean"]
    )
    difference_methods: List[Union[str, DifferenceReduction]] = field(
        default_factory=lambda: ["substractmatch", "ignorematch"]
    )
    normalization: List[bool] = field(default_factory=lambda: [False])
    feature_counts: List[int] = field(default_factory=lambda: [1, 4, 16, 32])
    prototype_init: List[str] = field(default_factory=lambda: ["uniform", "normal"])
    feature_init: List[str] = field(default_factory=lambda: ["uniform", "normal"])
    random_seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    epochs: int = 1000
    learning_rate: float = 0.1
    convergence_threshold: float = 1.0  # 100% accuracy
    
    @property
    def total_runs(self) -> int:
        """Calculate total number of experimental runs."""
        return (len(self.intersection_methods) * len(self.difference_methods) * 
                len(self.normalization) * len(self.feature_counts) * 
                len(self.prototype_init) * len(self.feature_init) * 
                len(self.random_seeds))


# Fast benchmark for development (24 runs, ~15 seconds)
# Uses xavier_uniform for reliable convergence during development
FAST_BENCHMARK_CONFIG = XORConfig(
    intersection_methods=["product", "mean", "max", "gmean"],
    difference_methods=["substractmatch", "ignorematch"], 
    normalization=[False],
    feature_counts=[4, 16],  # Focus on good feature counts
    prototype_init=["xavier_uniform"],  # Use working initialization
    feature_init=["xavier_uniform"],
    random_seeds=[0, 1, 2],
)

# Full paper replication (12,960 runs, ~2.2 hours)
# NOTE: Paper's "uniform" initialization may differ from PyTorch's implementation
# This may result in lower convergence rates than paper reports  
FULL_PAPER_CONFIG = XORConfig(
    intersection_methods=["min", "max", "product", "mean", "gmean", "softmin"],
    difference_methods=["ignorematch", "substractmatch"],
    normalization=[False, True],
    feature_counts=[1, 2, 4, 8, 16, 32],
    prototype_init=["uniform", "normal", "orthogonal"],
    feature_init=["uniform", "normal", "orthogonal"],
    random_seeds=list(range(9)),
)


@dataclass 
class XORResult:
    """Results from a single XOR training run."""
    
    intersection_method: str
    difference_method: str
    normalize: bool
    feature_count: int
    prototype_init: str
    feature_init: str
    seed: int
    
    final_loss: float
    final_accuracy: float
    converged: bool
    training_time: float
    
    # Optional detailed tracking
    loss_history: Optional[List[float]] = None
    accuracy_history: Optional[List[float]] = None


class XORBenchmark:
    """XOR benchmark runner for Tversky Neural Networks."""
    
    def __init__(self, config: XORConfig):
        self.config = config
        self.results: List[XORResult] = []
        
        # XOR dataset (matching working implementation)
        self.xor_inputs = torch.tensor([
            [0.0, 0.0],
            [0.0, 1.0], 
            [1.0, 0.0],
            [1.0, 1.0],
        ])
        self.xor_targets = torch.tensor([0, 1, 1, 0])
    
    def run_single_experiment(
        self, 
        intersection_method: str,
        difference_method: str,
        normalize: bool,
        feature_count: int,
        prototype_init: str,
        feature_init: str,
        seed: int,
        track_history: bool = False
    ) -> XORResult:
        """Run a single XOR training experiment."""
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        
        # Map initialization methods to what layer supports
        layer_proto_init = prototype_init if prototype_init != "orthogonal" else "xavier_uniform" 
        layer_feat_init = feature_init if feature_init != "orthogonal" else "xavier_uniform"
        
        # Create TverskyProjectionLayer
        layer = TverskyProjectionLayer(
            in_features=2,
            num_prototypes=2,  # XOR has 2 output classes
            num_features=feature_count,
            alpha=0.5,
            beta=0.5,
            learnable_ab=True,
            theta=1e-7,
            intersection_reduction=intersection_method,
            difference_reduction=difference_method,
            feature_init=layer_feat_init,
            prototype_init=layer_proto_init,
        )
        
        # Apply orthogonal initialization if requested (paper requirement)
        if prototype_init == "orthogonal":
            torch.nn.init.orthogonal_(layer.prototypes.data)
        if feature_init == "orthogonal":
            torch.nn.init.orthogonal_(layer.feature_bank.data)
        
        optimizer = optim.Adam(layer.parameters(), lr=self.config.learning_rate)
        
        # Training history tracking
        loss_history = [] if track_history else None
        accuracy_history = [] if track_history else None
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = layer(self.xor_inputs)
            loss = F.cross_entropy(outputs, self.xor_targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track history if requested
            if track_history:
                with torch.no_grad():
                    predicted = torch.argmax(outputs, dim=1)
                    accuracy = (predicted == self.xor_targets).float().mean().item()
                    loss_history.append(loss.item())
                    accuracy_history.append(accuracy)
        
        training_time = time.time() - start_time
        
        # Final evaluation
        with torch.no_grad():
            final_outputs = layer(self.xor_inputs)
            final_loss = F.cross_entropy(final_outputs, self.xor_targets).item()
            predicted = torch.argmax(final_outputs, dim=1)
            final_accuracy = (predicted == self.xor_targets).float().mean().item()
            converged = final_accuracy >= self.config.convergence_threshold
        
        return XORResult(
            intersection_method=intersection_method,
            difference_method=difference_method,
            normalize=normalize,
            feature_count=feature_count,
            prototype_init=prototype_init,
            feature_init=feature_init,
            seed=seed,
            final_loss=final_loss,
            final_accuracy=final_accuracy,
            converged=converged,
            training_time=training_time,
            loss_history=loss_history,
            accuracy_history=accuracy_history,
        )
    
    def run_benchmark(
        self, 
        verbose: bool = True,
        track_history: bool = False
    ) -> List[XORResult]:
        """Run complete benchmark suite."""
        
        total_runs = self.config.total_runs
        if verbose:
            print(f"Starting XOR benchmark: {total_runs} total runs")
            print(f"Estimated runtime: {total_runs * 0.6:.1f} seconds")
        
        results = []
        start_time = time.time()
        
        # Generate all parameter combinations
        combinations = product(
            self.config.intersection_methods,
            self.config.difference_methods,
            self.config.normalization,
            self.config.feature_counts,
            self.config.prototype_init,
            self.config.feature_init,
            self.config.random_seeds,
        )
        
        for i, (int_method, diff_method, normalize, n_features, 
                proto_init, feat_init, seed) in enumerate(combinations):
            
            if verbose and (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (total_runs - i - 1)
                print(f"  Progress: {i+1}/{total_runs} ({100*(i+1)/total_runs:.1f}%) "
                      f"ETA: {eta:.1f}s")
            
            try:
                result = self.run_single_experiment(
                    intersection_method=int_method,
                    difference_method=diff_method,
                    normalize=normalize,
                    feature_count=n_features,
                    prototype_init=proto_init,
                    feature_init=feat_init,
                    seed=seed,
                    track_history=track_history,
                )
                results.append(result)
                
            except Exception as e:
                if verbose:
                    print(f"  Warning: Run {i+1} failed: {e}")
                # Create failed result
                failed_result = XORResult(
                    intersection_method=int_method,
                    difference_method=diff_method,
                    normalize=normalize,
                    feature_count=n_features,
                    prototype_init=proto_init,
                    feature_init=feat_init,
                    seed=seed,
                    final_loss=float('nan'),
                    final_accuracy=0.5,  # Random guessing
                    converged=False,
                    training_time=0.0,
                )
                results.append(failed_result)
        
        total_time = time.time() - start_time
        if verbose:
            convergence_rate = sum(r.converged for r in results) / len(results)
            print(f"Benchmark complete: {total_time:.1f}s")
            print(f"Overall convergence rate: {convergence_rate:.2%}")
        
        self.results = results
        return results
    
    def analyze_results(self) -> Dict[str, float]:
        """Analyze benchmark results and compute convergence rates."""
        
        if not self.results:
            raise ValueError("No results to analyze. Run benchmark first.")
        
        analysis = {}
        
        # Overall convergence rate
        total_converged = sum(r.converged for r in self.results)
        analysis['overall_convergence_rate'] = total_converged / len(self.results)
        
        # By intersection method
        for method in self.config.intersection_methods:
            method_results = [r for r in self.results if r.intersection_method == method]
            if method_results:
                converged = sum(r.converged for r in method_results)
                analysis[f'convergence_rate_{method}'] = converged / len(method_results)
        
        # By difference method  
        for method in self.config.difference_methods:
            method_results = [r for r in self.results if r.difference_method == method]
            if method_results:
                converged = sum(r.converged for r in method_results)
                analysis[f'convergence_rate_{method}'] = converged / len(method_results)
        
        # By method combination (key paper finding)
        method_combos = set((r.intersection_method, r.difference_method) 
                           for r in self.results)
        for int_method, diff_method in method_combos:
            combo_results = [r for r in self.results 
                           if r.intersection_method == int_method 
                           and r.difference_method == diff_method]
            if combo_results:
                converged = sum(r.converged for r in combo_results)
                rate = converged / len(combo_results)
                analysis[f'convergence_rate_{int_method}_{diff_method}'] = rate
        
        return analysis


def run_fast_xor_benchmark(verbose: bool = True) -> Tuple[List[XORResult], Dict[str, float]]:
    """Run fast XOR benchmark for development (96 runs, ~60 seconds)."""
    
    benchmark = XORBenchmark(FAST_BENCHMARK_CONFIG)
    results = benchmark.run_benchmark(verbose=verbose)
    analysis = benchmark.analyze_results()
    
    if verbose:
        print("\n=== Fast XOR Benchmark Results ===")
        print(f"Total runs: {len(results)}")
        print(f"Overall convergence: {analysis['overall_convergence_rate']:.2%}")
        
        # Show key method combinations
        key_combos = [
            ('product', 'substractmatch'),
            ('mean', 'substractmatch'), 
            ('max', 'ignorematch'),
            ('gmean', 'ignorematch'),
        ]
        
        print("\nKey method combinations:")
        for int_method, diff_method in key_combos:
            key = f'convergence_rate_{int_method}_{diff_method}'
            if key in analysis:
                print(f"  {int_method} + {diff_method}: {analysis[key]:.2%}")
    
    return results, analysis


def run_full_xor_replication(verbose: bool = True) -> Tuple[List[XORResult], Dict[str, float]]:
    """Run full paper replication (12,960 runs, ~2.2 hours)."""
    
    if verbose:
        print("⚠️  WARNING: Full replication will take ~2.2 hours")
        print("Use run_fast_xor_benchmark() for development")
    
    benchmark = XORBenchmark(FULL_PAPER_CONFIG)
    results = benchmark.run_benchmark(verbose=verbose)
    analysis = benchmark.analyze_results()
    
    if verbose:
        print("\n=== Full XOR Replication Results ===")
        print(f"Total runs: {len(results)}")
        print(f"Overall convergence: {analysis['overall_convergence_rate']:.2%}")
        
        # Paper validation targets
        paper_targets = {
            ('product', 'substractmatch'): 0.53,
            ('mean', 'substractmatch'): 0.51,
            ('max', 'ignorematch'): 0.47,
            ('gmean', 'ignorematch'): 0.00,  # Should fail
        }
        
        print("\nPaper validation (expected vs actual):")
        for (int_method, diff_method), expected in paper_targets.items():
            key = f'convergence_rate_{int_method}_{diff_method}'
            if key in analysis:
                actual = analysis[key]
                diff = abs(actual - expected)
                status = "✅" if diff < 0.05 else "❌"
                print(f"  {int_method} + {diff_method}: {expected:.2%} vs {actual:.2%} {status}")
    
    return results, analysis