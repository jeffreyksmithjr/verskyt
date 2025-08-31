# Future Infrastructure Work

This document outlines pending infrastructure improvements for documentation, deployment, and build systems that are currently blocked by technical dependencies or repository status.

## Documentation Infrastructure

### MyST Parser Integration (Blocked)
**Current Issue**: MyST parser extensions are disabled in `docs/conf.py` due to compatibility issues with current Sphinx versions.

**Planned Work**:
- Enable MyST markdown parsing when Sphinx 8.x compatibility stabilizes
- Add Jupyter notebook rendering capability in documentation
- Convert existing markdown files to use MyST features

**Timeline**: Dependent on upstream Sphinx/MyST compatibility

### Enhanced Documentation Build
**Current Status**: Basic Sphinx documentation builds successfully locally

**Remaining Work**:
- Add automated documentation testing in CI
- Implement documentation link checking
- Add documentation build artifacts to GitHub Actions

## Deployment Infrastructure

### Read the Docs Integration (Blocked)
**Current Issue**: Repository is private, Read the Docs free tier requires public repositories

**Planned Work When Repository Goes Public**:
- Set up Read the Docs project integration
- Configure automatic builds on main branch pushes
- Update documentation URLs in `pyproject.toml`

**Alternative**: GitHub Pages workflow is documented in `docs/DEPLOYMENT.md:47-74`

### PyPI Publishing (Ready)
**Current Status**: Publishing workflow configured but not yet used

**Remaining Work**:
- Configure PyPI trusted publishing for first release
- Test publishing workflow with initial version tag
- Verify package installation from PyPI

## Build System Infrastructure

### Development Scripts Enhancement
**Current Status**: Basic scripts configured in `pyproject.toml:82-91`

**Potential Improvements**:
- Add script for local documentation serving with auto-reload
- Add script for running specific test categories
- Add script for package build verification

### CI/CD Optimization
**Current Status**: Two-tier CI strategy implemented (fast PR feedback, full main branch validation)

**Monitoring Needed**:
- Track CI performance and costs over time
- Adjust Python version matrix based on usage patterns
- Optimize test execution time if needed

## Current Blockers

1. **MyST Parser**: Waiting for Sphinx ecosystem compatibility
2. **Read the Docs**: Requires public repository
3. **PyPI Publishing**: Requires first version tag to test workflow

## No Action Required

This infrastructure work is either blocked by external dependencies or will be triggered by future actions (making repository public, creating first release tag). No immediate development work is needed.
