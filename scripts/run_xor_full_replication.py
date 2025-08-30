#!/usr/bin/env python3
"""
Full XOR Paper Replication Script

Reproduces the complete XOR experiment from:
"Tversky Neural Networks: Psychologically Plausible Deep Learning
with Differentiable Tversky Similarity" (Doumbouya et al., 2025)

Runs 12,960 experiments (~2.2 hours runtime)
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import torch

from verskyt.benchmarks.xor_suite import (
    FULL_PAPER_CONFIG,
    run_full_xor_replication,
)


def save_results(results, analysis, output_dir: Path):
    """Save results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save analysis summary
    analysis_file = output_dir / "xor_analysis.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to: {analysis_file}")

    # Save detailed results
    results_data = []
    for r in results:
        result_dict = {
            "intersection_method": r.intersection_method,
            "difference_method": r.difference_method,
            "normalize": r.normalize,
            "feature_count": r.feature_count,
            "prototype_init": r.prototype_init,
            "feature_init": r.feature_init,
            "seed": r.seed,
            "final_loss": r.final_loss,
            "final_accuracy": r.final_accuracy,
            "converged": r.converged,
            "training_time": r.training_time,
        }
        results_data.append(result_dict)

    results_file = output_dir / "xor_detailed_results.json"
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Detailed results saved to: {results_file}")


def print_paper_comparison(analysis: Dict[str, float]):
    """Print comparison with paper results."""
    print("\n" + "=" * 60)
    print("PAPER VALIDATION RESULTS")
    print("=" * 60)

    # Expected results from paper (Table in appendix_xor_results.tex)
    paper_targets = {
        "product + substractmatch": ("convergence_rate_product_substractmatch", 0.53),
        "mean + substractmatch": ("convergence_rate_mean_substractmatch", 0.51),
        "max + ignorematch": ("convergence_rate_max_ignorematch", 0.47),
        "max + substractmatch": ("convergence_rate_max_substractmatch", 0.44),
        "softmin + substractmatch": ("convergence_rate_softmin_substractmatch", 0.42),
        "min + ignorematch": ("convergence_rate_min_ignorematch", 0.42),
        "gmean + ignorematch": ("convergence_rate_gmean_ignorematch", 0.00),
        "gmean + substractmatch": ("convergence_rate_gmean_substractmatch", 0.00),
    }

    print(
        f"{'Method Combination':<25} {'Paper':<8} {'Actual':<8} {'Diff':<8} {'Status'}"
    )
    print("-" * 60)

    for display_name, (key, expected) in paper_targets.items():
        if key in analysis:
            actual = analysis[key]
            diff = abs(actual - expected)
            status = "✅ PASS" if diff < 0.05 else "❌ FAIL"
            print(
                f"{display_name:<25} {expected:<8.2%} "
                f"{actual:<8.2%} {diff:<8.2%} {status}"
            )
        else:
            print(
                f"{display_name:<25} {expected:<8.2%} {'N/A':<8} {'N/A':<8} {'⚠️ MISS'}"
            )

    print("\nValidation Criteria: ±5% tolerance")

    # Overall statistics
    print("\nOVERALL STATISTICS:")
    print(f"Total convergence rate: {analysis.get('overall_convergence_rate', 0):.2%}")
    print(f"Total experiments: {FULL_PAPER_CONFIG.total_runs:,}")


def print_method_rankings(analysis: Dict[str, float]):
    """Print method rankings by convergence rate."""
    print("\n" + "=" * 50)
    print("METHOD RANKINGS")
    print("=" * 50)

    # Extract method combination rates
    method_rates = []
    for key, rate in analysis.items():
        if (
            key.startswith("convergence_rate_") and "_" in key[17:]
        ):  # Skip single method rates
            method_name = key[17:].replace("_", " + ")
            method_rates.append((method_name, rate))

    # Sort by convergence rate (descending)
    method_rates.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Rank':<4} {'Method Combination':<25} {'Convergence Rate':<15}")
    print("-" * 50)

    for i, (method, rate) in enumerate(method_rates, 1):
        print(f"{i:<4} {method:<25} {rate:<15.2%}")


def main():
    parser = argparse.ArgumentParser(
        description="Run full XOR paper replication (12,960 experiments, ~2.2 hours)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/xor_replication"),
        help="Output directory for results (default: results/xor_replication)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save results to files"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument(
        "--device", default="auto", help="Device to use (cpu, cuda, mps, auto)"
    )

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print("XOR Full Paper Replication")
    print("=" * 50)
    print(f"Total experiments: {FULL_PAPER_CONFIG.total_runs:,}")
    print("Estimated runtime: ~2.2 hours")
    print(f"Device: {device}")
    print(f"Output directory: {args.output_dir}")

    if not args.quiet:
        print("\nConfiguration:")
        print(f"  Intersection methods: {FULL_PAPER_CONFIG.intersection_methods}")
        print(f"  Difference methods: {FULL_PAPER_CONFIG.difference_methods}")
        print(f"  Feature counts: {FULL_PAPER_CONFIG.feature_counts}")
        print(f"  Initialization methods: {FULL_PAPER_CONFIG.prototype_init}")
        print(f"  Random seeds: {len(FULL_PAPER_CONFIG.random_seeds)}")
        print(f"  Epochs per run: {FULL_PAPER_CONFIG.epochs}")

    # Confirm before starting
    response = input("\nThis will take ~2.2 hours. Continue? [y/N]: ")
    if response.lower() not in ["y", "yes"]:
        print("Aborted.")
        return

    # Set torch device
    if device != "cpu":
        torch.set_default_device(device)

    print(f"\nStarting full replication at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()

    try:
        # Run full replication
        results, analysis = run_full_xor_replication(verbose=not args.quiet)

        total_time = time.time() - start_time
        print(f"\nReplication completed in {total_time/3600:.2f} hours")

        # Print analysis
        print_paper_comparison(analysis)
        print_method_rankings(analysis)

        # Save results
        if not args.no_save:
            save_results(results, analysis, args.output_dir)

        print("\nReplication completed successfully!")

        # Validate key results
        key_validations = [
            ("product", "substractmatch", 0.53),
            ("gmean", "ignorematch", 0.00),
        ]
        all_valid = all(
            abs(analysis.get(f"convergence_rate_{m}_{d}", 0) - r) < 0.05
            for m, d, r in key_validations
        )
        status = "✅" if all_valid else "❌"
        print(f"Results validate paper findings: {status}")

    except KeyboardInterrupt:
        print(f"\nInterrupted after {(time.time() - start_time)/60:.1f} minutes")
        print("Partial results may be incomplete.")
    except Exception as e:
        print(f"\nError during replication: {e}")
        raise


if __name__ == "__main__":
    main()
