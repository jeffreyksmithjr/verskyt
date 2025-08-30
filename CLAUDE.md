# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Verskyt is a Python library implementing Tversky Neural Networks (TNNs) - psychologically plausible deep learning models based on differentiable Tversky similarity. The library provides PyTorch-based implementations of similarity computations and neural network layers.

## Development Commands

### Environment Setup
```bash
# Install in development mode with all dependencies
pip install -e ".[dev]"

# Install and activate pre-commit hooks (REQUIRED)
pre-commit install

# Verify setup
pre-commit run --all-files
```

### Quality Assurance (Automated via Pre-commit)
```bash
# Format code with black (88 char line length)
black verskyt tests

# Sort imports
isort verskyt tests

# Lint with flake8
flake8 verskyt tests

# Type checking
mypy verskyt

# Validate imports
python -c "import verskyt; print('All imports OK')"
```

### Testing
```bash
# Run all tests with coverage
pytest -v --cov=verskyt --cov-report=term-missing

# Run specific test file
pytest tests/test_basic_functionality.py -v

# Run specific test
pytest tests/test_basic_functionality.py::TestBasicSimilarity::test_similarity_shape -v

# Debug test failures
pytest tests/failing_test.py -vvs --pdb
```

### Pre-commit Validation
```bash
# Check all quality gates before committing
pre-commit run --all-files

# Bypass hooks only for emergencies (strongly discouraged)
git commit --no-verify
```

## Architecture

The library is organized into modular components:

- **verskyt/core/similarity.py**: Core mathematical operations for Tversky similarity
  - `tversky_similarity()`: Main similarity computation function
  - `compute_feature_membership()`: Feature membership scoring
  - `compute_salience()`: Salience weighting calculations
  - Enums for intersection/difference reduction methods

- **verskyt/layers/**: Neural network layers
  - `TverskyProjectionLayer`: Projects inputs to feature space
  - `TverskySimilarityLayer`: Computes similarity between inputs and prototypes

- **verskyt/utils/**: Utility functions
  - `initializers.py`: Custom weight initialization strategies

## Documentation

For detailed implementation guidance, see:
- **[Implementation Requirements](docs/requirements/tnn-specification.md)**: Complete mathematical specifications
- **[Implementation Plan](docs/implementation/plan.md)**: Development roadmap and testing strategy
- **[API Documentation](docs/api/)**: Detailed API reference

## Key Implementation Details

- Based on paper: "Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity" (Doumbouya et al., 2025)
- All operations are differentiable and PyTorch-compatible
- Similarity values are normalized to [0, 1] range
- Supports batch processing with shape conventions:
  - Objects/inputs: [batch_size, in_features]
  - Prototypes: [num_prototypes, in_features]
  - Features: [num_features, in_features]

## Testing Conventions

- Test files are in `tests/` directory following `test_*.py` naming
- Tests are organized by functionality (core, layers, utils)
- Use pytest fixtures for common test data
- Ensure all similarity outputs are in valid [0, 1] range
- Test gradient flow for all differentiable operations

## Behavioral Guidelines

### Core Review Principles
- **Verification-first approach**: Read and verify implementation before making claims about missing or incorrect code
- **Code as source of truth**: Use actual implementation lines as evidence for all assertions
- **Precise location references**: Include file paths and line numbers for all identified issues
- **Project priority alignment**: Frame feedback around correctness, documentation, testing, linting, and typing

### Communication Style
- Use neutral, technically grounded language focused on factual observations
- Report actual metrics and test results rather than subjective assessments
- Present both successful validations and identified issues
- Acknowledge limitations and uncertainty where they exist

### Language Patterns
- **Descriptive over evaluative**: "Implementation follows established patterns" not "excellent implementation"
- **Specific over general**: "87 tests passed, 3 failed" not "tests mostly passing"
- **Factual over emotional**: "Analysis complete" not "great job on the analysis"

### Code Analysis Methodology
1. **Read implementation thoroughly**: Examine actual code before making assertions
2. **Verify against specifications**: Compare implementation to documented requirements
3. **Validate test coverage**: Check that tests cover mathematical correctness and edge cases
4. **Assess code quality**: Review documentation, type hints, and adherence to style guidelines
5. **Provide actionable feedback**: Specify exact locations and clear improvement steps

### Review Output Format
```
## Code Analysis Summary
[Brief paragraph describing implementation approach and scope]

## Technical Validation
### ✅ Verified Implementation
- [Specific confirmations with file:line references]
- [Adherence to project standards]

### ❌ Issues Requiring Action  
- [Specific problems with file:line locations]
- [Clear remediation steps]

### 💡 Enhancement Opportunities
- [Minor improvements and suggestions]
```

### Quality Standards Focus
- **Documentation**: Verify docstrings reference relevant equations and provide clear usage examples
- **Testing**: Confirm adequate coverage of mathematical correctness, gradient flow, and edge cases
- **Type Safety**: Check for specific tensor shape annotations and correct type hints
- **Code Style**: Validate PEP 8 compliance and consistency with project patterns

### Reporting Format
```
Status: [Component] validation complete
Results: X/Y checks passed, Z warnings identified
Issues: [Specific problems with line numbers/locations]
Next: [Required actions, if any]
```

### Emoji Usage
- Limit to functional indicators only (⚠️ for warnings, ❌ for failures)
- Avoid celebratory emoji (🎉, 🚀, ✨) 
- Use sparingly and only when they add informational value

## Proactive Development Guidelines

### Pre-Development Checklist
Before writing code, always:
1. **Review specifications**: Check docs/requirements/tnn-specification.md for mathematical requirements
2. **Understand testing strategy**: Review docs/implementation/plan.md for testing approach  
3. **Set up environment**: Run `pip install -e ".[dev]" && pre-commit install`
4. **Validate setup**: Run `pre-commit run --all-files && pytest`

### Development Quality Gates

#### Before Each Commit
1. **Automated checks pass**: Pre-commit hooks handle formatting, linting, imports
2. **Tests validate actual functionality**: Not just setup - test claimed capabilities
3. **Imports properly exported**: Add new functions to appropriate `__init__.py` files
4. **Documentation complete**: Include tensor shapes, equation references

#### Before Creating PR
1. **Full test suite passes**: `pytest -v --cov=verskyt`
2. **Coverage maintained**: Core modules >80%, overall >60%
3. **No manual quality issues**: `black verskyt tests && isort verskyt tests && flake8 verskyt tests`
4. **Mathematical correctness**: Implementation matches paper specifications
5. **Integration validated**: New code works with existing PyTorch patterns

### Development Workflow Integration
```bash
# Recommended development flow
git checkout -b feature/my-feature

# Write code following TDD approach
pytest tests/test_new_feature.py -v  # Should fail initially
# ... implement feature ...
pytest tests/test_new_feature.py -v  # Should pass

# Pre-commit runs automatically
git add . && git commit -m "feat: implement new feature"

# Final validation before PR
pytest && echo "Ready for PR"
```

### Common Issue Prevention

**Import Consistency Issues:**
- Always update `__init__.py` when adding public functions
- Test imports: `python -c "from verskyt.module import new_function"`
- Run import validation: `pre-commit run validate-imports --all-files`

**Test Depth Issues:**
- Tests must validate actual capabilities, not just basic functionality
- XOR tests should verify learning capability, not just training steps
- Include negative tests (edge cases, expected failures)

**Code Quality Issues:**
- Pre-commit hooks prevent most formatting/linting issues
- Manual run: `pre-commit run --all-files` before pushing
- For urgent fixes: `git commit --no-verify` (document reason in commit message)

**Mathematical Implementation Issues:**
- Cross-reference all implementations with paper equations
- Include equation numbers in docstrings
- Test gradient flow for all differentiable operations
- Validate parameter bounds (α ≥ 0, β ≥ 0 per paper)