#!/usr/bin/env python3
"""
XOR Results Analysis Script

Analyzes XOR benchmark results and generates comparison reports.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import statistics


def load_results(results_file: Path) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def analyze_by_method_combination(results: List[Dict]) -> Dict[str, Dict]:
    """Analyze results by intersection/difference method combinations."""
    method_stats = {}
    
    for result in results:
        int_method = result['intersection_method']
        diff_method = result['difference_method']
        combo = f"{int_method}+{diff_method}"
        
        if combo not in method_stats:
            method_stats[combo] = {
                'convergence_rates': [],
                'final_accuracies': [],
                'final_losses': [],
                'training_times': [],
            }
        
        method_stats[combo]['convergence_rates'].append(1.0 if result['converged'] else 0.0)
        method_stats[combo]['final_accuracies'].append(result['final_accuracy'])
        if not (result['final_loss'] != result['final_loss']):  # Check for NaN
            method_stats[combo]['final_losses'].append(result['final_loss'])
        method_stats[combo]['training_times'].append(result['training_time'])
    
    # Compute statistics
    summary = {}
    for combo, stats in method_stats.items():
        summary[combo] = {
            'convergence_rate': statistics.mean(stats['convergence_rates']),
            'avg_final_accuracy': statistics.mean(stats['final_accuracies']),
            'avg_final_loss': statistics.mean(stats['final_losses']) if stats['final_losses'] else float('nan'),
            'avg_training_time': statistics.mean(stats['training_times']),
            'n_runs': len(stats['convergence_rates']),
        }
    
    return summary


def analyze_by_feature_count(results: List[Dict]) -> Dict[int, Dict]:
    """Analyze results by feature count."""
    feature_stats = {}
    
    for result in results:
        n_features = result['feature_count']
        
        if n_features not in feature_stats:
            feature_stats[n_features] = {
                'convergence_rates': [],
                'final_accuracies': [],
            }
        
        feature_stats[n_features]['convergence_rates'].append(1.0 if result['converged'] else 0.0)
        feature_stats[n_features]['final_accuracies'].append(result['final_accuracy'])
    
    # Compute statistics
    summary = {}
    for n_features, stats in feature_stats.items():
        summary[n_features] = {
            'convergence_rate': statistics.mean(stats['convergence_rates']),
            'avg_final_accuracy': statistics.mean(stats['final_accuracies']),
            'n_runs': len(stats['convergence_rates']),
        }
    
    return summary


def analyze_by_initialization(results: List[Dict]) -> Dict[str, Dict]:
    """Analyze results by initialization method."""
    init_stats = {}
    
    for result in results:
        proto_init = result['prototype_init']
        feature_init = result['feature_init']
        combo = f"{proto_init}+{feature_init}"
        
        if combo not in init_stats:
            init_stats[combo] = {
                'convergence_rates': [],
                'final_accuracies': [],
            }
        
        init_stats[combo]['convergence_rates'].append(1.0 if result['converged'] else 0.0)
        init_stats[combo]['final_accuracies'].append(result['final_accuracy'])
    
    # Compute statistics
    summary = {}
    for combo, stats in init_stats.items():
        summary[combo] = {
            'convergence_rate': statistics.mean(stats['convergence_rates']),
            'avg_final_accuracy': statistics.mean(stats['final_accuracies']),
            'n_runs': len(stats['convergence_rates']),
        }
    
    return summary


def print_method_analysis(method_summary: Dict[str, Dict]):
    """Print method combination analysis."""
    print("\n" + "="*60)
    print("ANALYSIS BY METHOD COMBINATION")
    print("="*60)
    
    # Sort by convergence rate
    sorted_methods = sorted(method_summary.items(), 
                          key=lambda x: x[1]['convergence_rate'], 
                          reverse=True)
    
    print(f"{'Method Combination':<25} {'Conv Rate':<10} {'Avg Acc':<10} {'Avg Loss':<10} {'N':<6}")
    print("-" * 60)
    
    for combo, stats in sorted_methods:
        print(f"{combo:<25} {stats['convergence_rate']:<10.2%} "
              f"{stats['avg_final_accuracy']:<10.3f} "
              f"{stats['avg_final_loss']:<10.3f} "
              f"{stats['n_runs']:<6}")


def print_feature_analysis(feature_summary: Dict[int, Dict]):
    """Print feature count analysis."""
    print("\n" + "="*40)
    print("ANALYSIS BY FEATURE COUNT")
    print("="*40)
    
    # Sort by feature count
    sorted_features = sorted(feature_summary.items())
    
    print(f"{'Features':<10} {'Conv Rate':<12} {'Avg Accuracy':<12} {'N':<6}")
    print("-" * 40)
    
    for n_features, stats in sorted_features:
        print(f"{n_features:<10} {stats['convergence_rate']:<12.2%} "
              f"{stats['avg_final_accuracy']:<12.3f} {stats['n_runs']:<6}")


def print_paper_comparison(method_summary: Dict[str, Dict]):
    """Compare with paper results."""
    print("\n" + "="*60)
    print("COMPARISON WITH PAPER RESULTS")
    print("="*60)
    
    # Paper targets from appendix_xor_results.tex
    paper_targets = {
        'product+substractmatch': 0.53,
        'mean+substractmatch': 0.51,
        'max+ignorematch': 0.47,
        'gmean+ignorematch': 0.00,
        'gmean+substractmatch': 0.00,
    }
    
    print(f"{'Method Combination':<25} {'Paper':<8} {'Actual':<8} {'Diff':<8} {'Status'}")
    print("-" * 58)
    
    for combo, paper_rate in paper_targets.items():
        if combo in method_summary:
            actual_rate = method_summary[combo]['convergence_rate']
            diff = abs(actual_rate - paper_rate)
            status = "✅" if diff < 0.05 else "❌"
            print(f"{combo:<25} {paper_rate:<8.2%} {actual_rate:<8.2%} {diff:<8.2%} {status}")
        else:
            print(f"{combo:<25} {paper_rate:<8.2%} {'N/A':<8} {'N/A':<8} ⚠️")


def main():
    parser = argparse.ArgumentParser(description="Analyze XOR benchmark results")
    parser.add_argument("results_file", type=Path, help="JSON file with detailed results")
    parser.add_argument("--paper-comparison", action="store_true", help="Include paper comparison")
    parser.add_argument("--output", type=Path, help="Save analysis to file")
    
    args = parser.parse_args()
    
    if not args.results_file.exists():
        print(f"Error: Results file {args.results_file} not found")
        return 1
    
    print(f"Analyzing results from: {args.results_file}")
    
    # Load results
    if args.results_file.name.endswith('_results.json'):
        # Detailed results file
        results = load_results(args.results_file)
        if isinstance(results, list):
            detailed_results = results
        else:
            detailed_results = results.get('results', [])
    else:
        # Analysis file
        data = load_results(args.results_file)
        print("Loaded analysis file - detailed breakdowns not available")
        for key, value in data.items():
            if key.startswith('convergence_rate_'):
                method = key[17:].replace('_', '+')
                print(f"{method}: {value:.2%}")
        return
    
    print(f"Total experiments: {len(detailed_results)}")
    
    # Perform analysis
    method_analysis = analyze_by_method_combination(detailed_results)
    feature_analysis = analyze_by_feature_count(detailed_results)
    init_analysis = analyze_by_initialization(detailed_results)
    
    # Print results
    print_method_analysis(method_analysis)
    print_feature_analysis(feature_analysis)
    
    if args.paper_comparison:
        print_paper_comparison(method_analysis)
    
    # Save analysis if requested
    if args.output:
        analysis_data = {
            'method_combinations': method_analysis,
            'feature_counts': feature_analysis,
            'initialization_methods': init_analysis,
            'total_experiments': len(detailed_results),
        }
        
        with open(args.output, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        print(f"\nAnalysis saved to: {args.output}")


if __name__ == "__main__":
    main()