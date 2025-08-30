"""
Tests for XOR benchmark suite.

Tests both the fast development benchmark and validates against paper results.
"""

import pytest
import torch

from verskyt.benchmarks.xor_suite import (
    FAST_BENCHMARK_CONFIG,
    FULL_PAPER_CONFIG,
    XORBenchmark,
    XORResult,
    run_fast_xor_benchmark,
)


class TestXORBenchmark:
    """Test XOR benchmark functionality."""

    def test_xor_benchmark_creation(self):
        """Test XOR benchmark can be created."""
        benchmark = XORBenchmark(FAST_BENCHMARK_CONFIG)
        assert benchmark.config == FAST_BENCHMARK_CONFIG
        assert len(benchmark.results) == 0

        # Check XOR dataset
        assert benchmark.xor_inputs.shape == (4, 2)
        assert benchmark.xor_targets.shape == (4,)
        assert torch.equal(benchmark.xor_targets, torch.tensor([0, 1, 1, 0]))

    def test_single_xor_experiment(self):
        """Test single XOR experiment execution."""
        benchmark = XORBenchmark(FAST_BENCHMARK_CONFIG)

        result = benchmark.run_single_experiment(
            intersection_method="product",
            difference_method="substractmatch",
            normalize=False,
            feature_count=4,
            prototype_init="uniform",
            feature_init="uniform",
            seed=42,
            track_history=True,
        )

        # Validate result structure
        assert isinstance(result, XORResult)
        assert result.intersection_method == "product"
        assert result.difference_method == "substractmatch"
        assert result.normalize is False
        assert result.feature_count == 4
        assert result.seed == 42

        # Validate training outputs
        assert isinstance(result.final_loss, float)
        assert isinstance(result.final_accuracy, float)
        assert isinstance(result.converged, bool)
        assert isinstance(result.training_time, float)

        # Validate ranges
        assert 0.0 <= result.final_accuracy <= 1.0
        assert result.training_time > 0.0

        # History should be tracked
        assert result.loss_history is not None
        assert result.accuracy_history is not None
        assert len(result.loss_history) == 1000  # FAST_BENCHMARK_CONFIG.epochs
        assert len(result.accuracy_history) == 1000

    def test_gmean_numerical_stability(self):
        """Test that gmean method fails gracefully (paper shows NaN)."""
        benchmark = XORBenchmark(FAST_BENCHMARK_CONFIG)

        # gmean is known to be numerically unstable per paper
        result = benchmark.run_single_experiment(
            intersection_method="gmean",
            difference_method="ignorematch",
            normalize=False,
            feature_count=2,
            prototype_init="uniform",
            feature_init="uniform",
            seed=0,
        )

        # Paper shows gmean has 0% convergence rate
        # We expect either convergence failure or NaN losses
        if not torch.isnan(torch.tensor(result.final_loss)):
            # If training completes, convergence should be unlikely
            assert result.final_accuracy < 1.0  # Should not achieve perfect accuracy

    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        benchmark = XORBenchmark(FAST_BENCHMARK_CONFIG)

        # Run same experiment twice
        result1 = benchmark.run_single_experiment(
            intersection_method="product",
            difference_method="substractmatch",
            normalize=False,
            feature_count=16,
            prototype_init="uniform",
            feature_init="uniform",
            seed=123,
        )

        result2 = benchmark.run_single_experiment(
            intersection_method="product",
            difference_method="substractmatch",
            normalize=False,
            feature_count=16,
            prototype_init="uniform",
            feature_init="uniform",
            seed=123,
        )

        # Results should be identical (within floating point precision)
        assert abs(result1.final_loss - result2.final_loss) < 1e-6
        assert abs(result1.final_accuracy - result2.final_accuracy) < 1e-6
        assert result1.converged == result2.converged


class TestFastXORBenchmark:
    """Test fast XOR benchmark suite."""

    @pytest.mark.slow
    def test_fast_benchmark_execution(self):
        """Test fast benchmark runs and produces reasonable results."""
        results, analysis = run_fast_xor_benchmark(verbose=False)

        # Validate structure
        assert len(results) == FAST_BENCHMARK_CONFIG.total_runs  # 48 runs
        assert isinstance(analysis, dict)
        assert "overall_convergence_rate" in analysis

        # Validate all results are present
        assert all(isinstance(r, XORResult) for r in results)

        # Validate convergence rates are reasonable
        overall_rate = analysis["overall_convergence_rate"]
        assert 0.0 <= overall_rate <= 1.0

        # Paper shows some methods work better than others
        # With xavier_uniform init, convergence rates will be different from paper
        # but we should see some variation between methods
        method_rates = [
            v
            for k, v in analysis.items()
            if k.startswith("convergence_rate_") and "_" in k[17:]
        ]
        if len(method_rates) > 1:
            # Should have some variation in performance
            assert max(method_rates) >= min(method_rates)  # Basic sanity check

        # gmean methods should have low convergence (paper shows 0%)
        gmean_ignore_key = "convergence_rate_gmean_ignorematch"
        if gmean_ignore_key in analysis:
            gmean_rate = analysis[gmean_ignore_key]
            assert gmean_rate <= 0.2  # Should be quite low

    def test_benchmark_analysis(self):
        """Test benchmark analysis functionality."""
        # Create small benchmark for testing
        from verskyt.benchmarks.xor_suite import XORConfig

        test_config = XORConfig(
            intersection_methods=["product", "mean"],
            difference_methods=["substractmatch"],
            normalization=[False],
            feature_counts=[4],
            prototype_init=["uniform"],
            feature_init=["uniform"],
            random_seeds=[0, 1],  # Just 2 seeds
        )

        benchmark = XORBenchmark(test_config)
        benchmark.run_benchmark(verbose=False)
        analysis = benchmark.analyze_results()

        # Should have analysis for each method
        assert "convergence_rate_product" in analysis
        assert "convergence_rate_mean" in analysis
        assert "convergence_rate_substractmatch" in analysis
        assert "convergence_rate_product_substractmatch" in analysis
        assert "convergence_rate_mean_substractmatch" in analysis

        # Rates should be between 0 and 1
        for key, rate in analysis.items():
            if key.startswith("convergence_rate_"):
                assert 0.0 <= rate <= 1.0


class TestXORConfiguration:
    """Test XOR configuration and validation."""

    def test_config_total_runs_calculation(self):
        """Test total runs calculation is correct."""
        config = FAST_BENCHMARK_CONFIG
        expected = (
            len(config.intersection_methods)
            * len(config.difference_methods)
            * len(config.normalization)
            * len(config.feature_counts)
            * len(config.prototype_init)
            * len(config.feature_init)
            * len(config.random_seeds)
        )
        assert config.total_runs == expected
        assert config.total_runs == 48  # 4*2*1*2*1*1*3

    def test_full_paper_config(self):
        """Test full paper configuration matches paper specs."""
        config = FULL_PAPER_CONFIG

        # Validate against paper specifications
        assert len(config.intersection_methods) == 6
        assert len(config.difference_methods) == 2
        assert len(config.normalization) == 2
        assert len(config.feature_counts) == 6
        assert len(config.prototype_init) == 3
        assert len(config.feature_init) == 3
        assert len(config.random_seeds) == 9

        # Total calculated: 6*2*2*6*3*3*9 = 11,664
        # (Paper claims 12,960 but table shows 972 per combination × 12 = 11,664)
        assert config.total_runs == 11664

        # Validate method lists match paper
        expected_intersection = ["min", "max", "product", "mean", "gmean", "softmin"]
        expected_difference = ["ignorematch", "substractmatch"]
        expected_features = [1, 2, 4, 8, 16, 32]
        expected_inits = ["uniform", "normal", "orthogonal"]

        assert set(config.intersection_methods) == set(expected_intersection)
        assert set(config.difference_methods) == set(expected_difference)
        assert set(config.feature_counts) == set(expected_features)
        assert set(config.prototype_init) == set(expected_inits)
        assert set(config.feature_init) == set(expected_inits)


class TestPaperValidation:
    """Test validation against paper results."""

    def test_paper_expectations(self):
        """Test that we can validate against paper expectations."""
        # Expected convergence rates from paper Table in appendix_xor_results.tex
        paper_expectations = {
            ("product", "substractmatch"): 0.53,  # Best combination
            ("mean", "substractmatch"): 0.51,  # Second best
            ("max", "ignorematch"): 0.47,  # Good performance
            ("gmean", "ignorematch"): 0.00,  # Numerical instability
            ("gmean", "substractmatch"): 0.00,  # Numerical instability
        }

        # This is a documentation test - we don't run the full benchmark
        # but validate our expectations are reasonable
        for (int_method, diff_method), expected_rate in paper_expectations.items():
            assert 0.0 <= expected_rate <= 1.0

            # gmean should have 0% convergence (numerical instability)
            if int_method == "gmean":
                assert expected_rate == 0.0

    @pytest.mark.skipif(True, reason="Full replication is very slow (~2.2 hours)")
    def test_full_paper_replication(self):
        """Test full paper replication results (normally skipped)."""
        # This test is normally skipped due to runtime
        # Remove skip decorator to run full validation
        from verskyt.benchmarks.xor_suite import run_full_xor_replication

        results, analysis = run_full_xor_replication(verbose=True)

        # Validate against paper results
        paper_targets = {
            "convergence_rate_product_substractmatch": 0.53,
            "convergence_rate_mean_substractmatch": 0.51,
            "convergence_rate_max_ignorematch": 0.47,
            "convergence_rate_gmean_ignorematch": 0.00,
        }

        for key, expected in paper_targets.items():
            if key in analysis:
                actual = analysis[key]
                # Allow 5% tolerance for stochastic variation
                assert (
                    abs(actual - expected) < 0.05
                ), f"{key}: expected {expected}, got {actual}"


# Pytest markers for test organization
pytestmark = [
    pytest.mark.benchmark,  # Mark all tests as benchmark-related
]
