# Development Guide

This guide covers the development workflow, quality standards, and contribution process for Verskyt.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/verskyt.git
cd verskyt

# Set up development environment
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black verskyt tests
isort verskyt tests
```

## Development Environment Setup

### 1. Install Dependencies

```bash
# Install with all development tools
pip install -e ".[dev]"

# Or install specific tool sets
pip install -e ".[dev,visualization,benchmarks]"
```

### 2. Pre-commit Hooks

Pre-commit hooks automatically enforce code quality standards:

```bash
# Install hooks (one-time setup)
pre-commit install

# Manually run all hooks
pre-commit run --all-files

# Skip hooks for emergency commits (discouraged)
git commit --no-verify
```

**Hooks include:**
- **Black**: Code formatting (88 char line length)
- **isort**: Import organization
- **flake8**: Linting and style checking  
- **Import validation**: Ensures all imports resolve correctly
- **Basic tests**: Run core test suite on push

### 3. Code Quality Standards

#### Formatting
- **Line length**: 88 characters (Black default)
- **Import organization**: Use isort with Black profile
- **Docstrings**: Include tensor shapes and equation references
- **Type hints**: Required for all public functions

#### Testing Requirements
- **Coverage**: Minimum 60% overall, 75% for core modules
- **Test categories**: Unit tests, integration tests, gradient flow validation
- **Naming**: Test files `test_*.py`, test functions `test_*`

#### Documentation Standards
- **Docstrings**: Include Args, Returns, tensor shapes
- **Examples**: Provide usage examples for public APIs
- **Equation references**: Link to paper equations where applicable

#### CI/CD Integration
- **GitHub Actions**: Automated quality checks on all PRs and pushes
- **Quality gates**: Code formatting, linting, import validation, testing
- **Coverage reporting**: Automatic coverage tracking via Codecov
- **Multi-Python support**: Tests run on Python 3.8, 3.9, 3.10, 3.11

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes
# ... edit code ...

# Pre-commit hooks run automatically
git add .
git commit -m "feat: add your feature description"

# Push and create PR
git push origin feature/your-feature-name
```

### 2. Quality Checklist (Pre-PR)

**Code Quality:**
- [ ] Code formatted with Black (pre-commit enforces this)
- [ ] Imports organized with isort  
- [ ] All flake8 checks pass
- [ ] Type hints added for public functions
- [ ] No import errors or missing exports

**Testing:**
- [ ] All existing tests pass: `pytest`
- [ ] New tests written for new functionality
- [ ] Test coverage maintained: `pytest --cov=verskyt`
- [ ] Gradient flow tested for differentiable operations

**Documentation:**
- [ ] Docstrings include tensor shapes
- [ ] Public API changes documented
- [ ] README updated if needed

**Mathematical Correctness:**
- [ ] Implementation matches paper specifications
- [ ] Parameter validation includes bounds checking
- [ ] Numerical stability considered

### 3. Common Issues & Solutions

**Pre-commit failures:**
```bash
# Fix formatting issues
black verskyt tests
isort verskyt tests

# Fix import issues
# Check that imports match exports in __init__.py files

# Fix test failures
pytest -x  # Stop on first failure for debugging
```

**Import validation failures:**
```bash
# Ensure functions are exported in __init__.py
# Example: Add to verskyt/core/__init__.py:
from verskyt.core.similarity import your_new_function
__all__ = [..., "your_new_function"]
```

## Testing Strategy

### Test Organization
- `tests/test_basic_functionality.py`: Layer functionality and integration
- `tests/test_core_similarity.py`: Mathematical correctness
- Additional test files by module

### Test Categories

1. **Mathematical Correctness**
   - Similarity values in [0,1] range
   - Asymmetry property validation
   - Intersection/difference method verification

2. **Gradient Flow**
   - Backpropagation through all operations
   - Parameter learning validation
   - Numerical gradient checking

3. **Integration Testing**
   - PyTorch compatibility
   - Training loop functionality
   - XOR learning capability (validates non-linearity)

4. **Edge Cases**
   - Zero vectors, large values
   - Numerical stability
   - Batch processing

### Running Tests

```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_core_similarity.py -v

# Run specific test
pytest tests/test_core_similarity.py::TestTverskySimilarity::test_asymmetry -v

# Debug failing test
pytest tests/test_failing.py -vvs --pdb
```

## Code Review Process

### For Authors
1. Ensure all pre-commit hooks pass
2. Run full test suite locally
3. Complete quality checklist above
4. Write descriptive PR description with:
   - Summary of changes
   - Testing performed
   - Breaking changes (if any)

### For Reviewers
1. Verify mathematical correctness against paper
2. Check test coverage for new functionality
3. Validate code follows project patterns
4. Ensure documentation is complete

## Release Process

1. **Version bump** in `pyproject.toml`
2. **Update CHANGELOG.md** with new features/fixes
3. **Tag release**: `git tag v0.2.0`
4. **Build and publish**: Automated via CI/CD

## CI/CD Monitoring

### GitHub Actions Workflows

**Main CI Pipeline (`.github/workflows/ci.yml`):**
- **Quality checks**: Black, isort, flake8, import validation
- **Multi-Python testing**: Tests on Python 3.8-3.11
- **Coverage validation**: Enforces 60% overall, 75% core module coverage
- **Integration tests**: XOR learning validation, package import testing
- **Documentation checks**: Validates required docs are present

**Pre-commit Pipeline (`.github/workflows/pre-commit.yml`):**
- Fast feedback on basic quality issues
- Runs same hooks as local pre-commit
- Skips slow tests for quick iteration

### Monitoring CI Results

**Success Indicators:**
- ✅ All workflow jobs pass
- ✅ Coverage thresholds met
- ✅ Integration tests validate functionality
- ✅ No import or formatting issues

**Handling CI Failures:**

```bash
# Reproduce CI failures locally
pip install -e ".[dev]"
pre-commit run --all-files  # Check formatting/linting
pytest -v --cov=verskyt     # Check tests and coverage

# Fix common CI issues
black verskyt tests         # Fix formatting
isort verskyt tests         # Fix imports
pytest -x                   # Debug test failures
```

**Coverage Failures:**
```bash
# Check coverage details
pytest --cov=verskyt --cov-report=html
# Opens htmlcov/index.html for detailed coverage report

# Focus on uncovered core modules
pytest --cov=verskyt.core --cov-report=term-missing
```

## Troubleshooting

### Pre-commit Issues
```bash
# Reset hooks if corrupted
pre-commit uninstall
pre-commit install

# Update hook versions
pre-commit autoupdate
```

### Development Environment Issues
```bash
# Reinstall in development mode
pip uninstall verskyt
pip install -e ".[dev]"

# Clear pytest cache
rm -rf .pytest_cache __pycache__
```

### CI-Specific Issues
```bash
# Test Python version compatibility locally with pyenv
pyenv install 3.8.18 3.9.18 3.10.13 3.11.7
pyenv local 3.8.18
pip install -e ".[dev]" && pytest  # Test each version

# Debug import validation failures
python -c "import verskyt; print('Import successful')"

# Check GitHub Actions logs
# Visit: https://github.com/your-repo/verskyt/actions
```

### Performance Issues
```bash
# Profile code
python -m cProfile -s tottime your_script.py

# Memory profiling
pip install memory-profiler
python -m memory_profiler your_script.py
```