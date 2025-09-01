"""
Tests for example scripts to ensure they run without errors.

This module provides automated testing for all example scripts to prevent
regressions and ensure examples remain functional as the library evolves.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

# Configuration for example testing
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
DEFAULT_TIMEOUT = 300  # 5 minutes default timeout
EXAMPLE_CONFIGS = {
    # Configure specific examples with custom timeouts or skip conditions
    "visualization_demo.py": {"timeout": 120, "requires": ["visualization"]},
    "research_tutorial.py": {"timeout": 300, "requires": []},
    "intervention_demo.py": {"timeout": 180, "requires": []},
}


def discover_examples() -> List[Path]:
    """Discover all Python example files."""
    if not EXAMPLES_DIR.exists():
        return []

    examples = []
    for example_file in EXAMPLES_DIR.glob("*.py"):
        # Skip __init__.py and other non-example files
        if example_file.name.startswith("__"):
            continue
        examples.append(example_file)

    return sorted(examples)


def check_example_requirements(example_name: str) -> Tuple[bool, Optional[str]]:
    """Check if example requirements are met."""
    config = EXAMPLE_CONFIGS.get(example_name, {})
    requirements = config.get("requires", [])

    for requirement in requirements:
        if requirement == "visualization":
            try:
                import matplotlib
                import seaborn
                import sklearn

                matplotlib.use("Agg")  # Use non-interactive backend
            except ImportError as e:
                return False, f"Missing visualization dependencies: {e}"

    return True, None


def run_example(
    example_path: Path, timeout: int = DEFAULT_TIMEOUT
) -> Tuple[bool, str, str]:
    """
    Run an example script and return success status, stdout, and stderr.

    Args:
        example_path: Path to the example script
        timeout: Maximum time to wait for completion (seconds)

    Returns:
        Tuple of (success, stdout, stderr)
    """
    try:
        # Run the example in a subprocess
        result = subprocess.run(
            [sys.executable, str(example_path)],
            cwd=example_path.parent.parent,  # Run from project root
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        success = result.returncode == 0
        return success, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        return False, "", f"Example timed out after {timeout} seconds"
    except Exception as e:
        return False, "", f"Failed to run example: {e}"


@pytest.mark.examples
class TestExamples:
    """Test class for example script validation."""

    @pytest.mark.parametrize("example_path", discover_examples())
    def test_example_runs_successfully(self, example_path: Path):
        """Test that example script runs without errors."""
        example_name = example_path.name

        # Check if example requirements are met
        requirements_met, skip_reason = check_example_requirements(example_name)
        if not requirements_met:
            pytest.skip(f"Skipping {example_name}: {skip_reason}")

        # Get example-specific configuration
        config = EXAMPLE_CONFIGS.get(example_name, {})
        timeout = config.get("timeout", DEFAULT_TIMEOUT)

        # Run the example
        success, stdout, stderr = run_example(example_path, timeout)

        # Provide detailed information on failure
        if not success:
            failure_info = f"""
Example {example_name} failed to run successfully.

STDOUT:
{stdout}

STDERR:
{stderr}

Example path: {example_path}
Timeout: {timeout}s
"""
            pytest.fail(failure_info)

        # Basic output validation
        assert len(stdout) > 0, f"Example {example_name} produced no output"

        # Check for common error indicators in output
        error_indicators = ["Traceback", "Error:", "Exception:", "failed"]
        stdout_lower = stdout.lower()
        for indicator in error_indicators:
            if (
                indicator.lower() in stdout_lower
                and "error" not in example_name.lower()
            ):
                pytest.fail(
                    f"Example {example_name} output contains error indicator: {indicator}"
                )


@pytest.mark.examples
class TestExampleIntegration:
    """Integration tests for example functionality."""

    def test_all_examples_discoverable(self):
        """Test that example discovery works correctly."""
        examples = discover_examples()
        assert len(examples) > 0, "No examples found"

        # Check that known examples are discovered
        example_names = [ex.name for ex in examples]
        expected_examples = ["research_tutorial.py", "visualization_demo.py"]

        for expected in expected_examples:
            assert expected in example_names, f"Expected example {expected} not found"

    def test_example_directory_structure(self):
        """Test that examples directory has proper structure."""
        assert (
            EXAMPLES_DIR.exists()
        ), f"Examples directory {EXAMPLES_DIR} does not exist"
        assert EXAMPLES_DIR.is_dir(), f"Examples path {EXAMPLES_DIR} is not a directory"

        # Check for README
        readme_path = EXAMPLES_DIR / "README.md"
        if readme_path.exists():
            content = readme_path.read_text()
            assert len(content) > 0, "Examples README is empty"


@pytest.mark.slow
@pytest.mark.examples
class TestExamplePerformance:
    """Performance-related tests for examples (marked as slow)."""

    def test_examples_complete_within_timeout(self):
        """Test that examples complete within reasonable time limits."""
        examples = discover_examples()

        performance_results = {}
        for example_path in examples:
            example_name = example_path.name

            # Skip if requirements not met
            requirements_met, _ = check_example_requirements(example_name)
            if not requirements_met:
                continue

            # Measure execution time
            start_time = time.time()
            config = EXAMPLE_CONFIGS.get(example_name, {})
            timeout = config.get("timeout", DEFAULT_TIMEOUT)

            success, _, _ = run_example(example_path, timeout)
            end_time = time.time()

            if success:
                execution_time = end_time - start_time
                performance_results[example_name] = execution_time

                # Check if example runs within expected time
                expected_time = timeout * 0.8  # Should complete in 80% of timeout
                assert execution_time < expected_time, (
                    f"Example {example_name} took {execution_time:.1f}s, "
                    f"expected < {expected_time:.1f}s"
                )

        # Ensure we tested at least some examples
        assert len(performance_results) > 0, "No examples were performance tested"


# Utility functions for CI integration
def validate_all_examples() -> bool:
    """
    Validate all examples and return overall success status.

    This function can be called from CI scripts or other automation.
    """
    examples = discover_examples()
    if not examples:
        print("No examples found to validate")
        return False

    success_count = 0
    total_count = len(examples)

    for example_path in examples:
        example_name = example_path.name
        print(f"Testing {example_name}...", end=" ")

        # Check requirements
        requirements_met, skip_reason = check_example_requirements(example_name)
        if not requirements_met:
            print(f"SKIPPED ({skip_reason})")
            total_count -= 1
            continue

        # Run example
        config = EXAMPLE_CONFIGS.get(example_name, {})
        timeout = config.get("timeout", DEFAULT_TIMEOUT)
        success, stdout, stderr = run_example(example_path, timeout)

        if success:
            print("PASSED")
            success_count += 1
        else:
            print("FAILED")
            print(f"  STDERR: {stderr}")

    print(f"\nResults: {success_count}/{total_count} examples passed")
    return success_count == total_count


if __name__ == "__main__":
    # Allow running this test module directly
    success = validate_all_examples()
    sys.exit(0 if success else 1)
