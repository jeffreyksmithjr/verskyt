#!/usr/bin/env python3
"""
Fast XOR Benchmark Script

Runs a subset of XOR experiments for development and validation.
96 experiments (~60 seconds runtime)
"""

import argparse
import json
import time
from pathlib import Path

import torch

from verskyt.benchmarks.xor_suite import run_fast_xor_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Run fast XOR benchmark (96 experiments, ~60 seconds)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/xor_fast"),
        help="Output directory for results (default: results/xor_fast)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    print("XOR Fast Benchmark")
    print("=" * 30)
    print("Total experiments: 96")
    print("Estimated runtime: ~60 seconds")
    print(f"Output directory: {args.output_dir}")
    
    start_time = time.time()
    
    try:
        # Run fast benchmark
        results, analysis = run_fast_xor_benchmark(verbose=not args.quiet)
        
        total_time = time.time() - start_time
        print(f"\nBenchmark completed in {total_time:.1f} seconds")
        
        # Save results if requested
        if not args.no_save:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save analysis
            analysis_file = args.output_dir / "fast_analysis.json" 
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"Analysis saved to: {analysis_file}")
        
        # Quick validation
        overall_rate = analysis.get('overall_convergence_rate', 0)
        product_rate = analysis.get('convergence_rate_product_substractmatch', 0)
        gmean_rate = analysis.get('convergence_rate_gmean_ignorematch', 1)  # Default high to detect failure
        
        print(f"\nQuick Validation:")
        print(f"  Overall convergence: {overall_rate:.2%}")
        print(f"  Best method (product+substractmatch): {product_rate:.2%}")  
        print(f"  Problem method (gmean+ignorematch): {gmean_rate:.2%}")
        
        # Basic validation
        validation_pass = (
            overall_rate > 0.1 and  # Some methods should work
            product_rate >= overall_rate * 0.8 and  # Best method should be above average
            gmean_rate < 0.3  # gmean should struggle
        )
        
        print(f"  Status: {'✅ PASS' if validation_pass else '❌ FAIL'}")
        
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        raise


if __name__ == "__main__":
    main()