# Feature Documentation Checklist

Use this checklist to ensure complete documentation for any new feature before merging.

## Code-Level Documentation

### Google Style Docstrings (Required)
- [ ] All public functions have comprehensive docstrings with:
  - [ ] **Args**: All parameters with types and tensor shapes where applicable
  - [ ] **Returns**: Return type and description with shapes for tensors  
  - [ ] **Raises**: Exceptions that may be raised
  - [ ] **Note**: Implementation details, mathematical context
  - [ ] **Example**: Usage examples for complex functions

- [ ] All public classes have comprehensive docstrings with:
  - [ ] **Attributes**: All public attributes with types and descriptions
  - [ ] **Note**: Implementation details and usage patterns
  - [ ] **Example**: Basic usage example in class docstring

### Mathematical Context (For Core Functions)
- [ ] Reference paper equations where applicable (e.g., "Implements Equation 3 from Doumbouya et al., 2025")
- [ ] Include mathematical notation and variable definitions
- [ ] Explain relationship to Tversky similarity theory

### Type Hints and Shapes
- [ ] All function parameters have type hints
- [ ] Tensor parameters include shape information (e.g., `torch.Tensor[batch_size, features]`)
- [ ] Return types are properly annotated

## API Documentation Infrastructure

### Sphinx Documentation Files
- [ ] Created `docs/api/module_name.rst` with proper autodoc directives
- [ ] Created `docs/api/module_name.md` with eval-rst blocks
- [ ] Added module to `docs/api/index.rst` toctree
- [ ] Added module to `docs/api/index.md` toctree

### Documentation Build Validation
- [ ] Sphinx can build documentation without warnings
- [ ] All classes and functions appear in generated docs
- [ ] Cross-references work correctly

## Integration Examples

### Examples Directory
- [ ] Created or updated example in `examples/` directory
- [ ] Example follows established patterns from `examples/research_tutorial.py`
- [ ] Example demonstrates integration with existing functionality
- [ ] Example includes comments explaining key concepts

### README Updates (If Applicable)
- [ ] Updated main README.md for major new functionality
- [ ] Added to "Quick Reference" or "Main Classes" sections if appropriate
- [ ] Updated import examples if new public APIs added

## Quality Validation

### Pre-Commit Compliance
- [ ] All pre-commit hooks pass (`pre-commit run --all-files`)
- [ ] Black formatting applied
- [ ] Import sorting (isort) applied
- [ ] Flake8 linting passes
- [ ] No trailing whitespace or end-of-file issues

### Import Validation
- [ ] New functions/classes added to appropriate `__init__.py` files
- [ ] Import validation passes: `python -c "from verskyt.module import new_function"`
- [ ] Public API is properly exported

### Test Integration
- [ ] Documentation examples can be run as code
- [ ] Docstring examples are valid Python code
- [ ] Examples produce expected outputs

## Final Review

### Documentation Completeness
- [ ] Feature is fully documented at code level
- [ ] API reference documentation is complete
- [ ] Integration examples demonstrate real usage
- [ ] No "TODO" or placeholder text remains

### Consistency with Project Standards
- [ ] Documentation style matches existing modules
- [ ] Mathematical notation consistent with paper
- [ ] Code examples follow project conventions
- [ ] Error messages and logging are appropriate

---

## Usage Notes

1. **Copy this checklist** for each new feature or module
2. **Check off items** as you complete them
3. **Include completed checklist** in PR description
4. **Reviewer validates** that all items are properly completed

This checklist ensures that documentation is treated as a first-class requirement for all features, not an afterthought.