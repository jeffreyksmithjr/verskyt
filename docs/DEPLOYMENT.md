# Deployment Guide

This guide covers deploying Verskyt documentation and Python packages.

## 📚 Documentation Deployment

### Read the Docs (Recommended)

1. **Setup Read the Docs**:
   - Go to https://readthedocs.org/
   - Sign in with your GitHub account
   - Click "Import a Project"
   - Select the `verskyt` repository
   - Read the Docs will auto-detect the Sphinx configuration

2. **Configuration**:
   - Read the Docs will automatically:
     - Find `docs/conf.py`
     - Install dependencies from `pyproject.toml[dev]`
     - Build the documentation on every push to `main`
   
3. **Access**:
   - Your docs will be available at: https://verskyt.readthedocs.io/
   - Updates automatically on every push to main branch

### Alternative: GitHub Pages

If you prefer GitHub Pages, add this workflow to `.github/workflows/docs.yml`:

```yaml
name: Deploy Documentation
on:
  push:
    branches: [ main ]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: pip install -e ".[dev]"
    - name: Build docs
      run: cd docs && make html
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build/html
```

Then enable GitHub Pages in repository settings.

## 📦 Package Deployment (PyPI)

### One-Time Setup

1. **Configure PyPI Trusted Publishing**:
   - Go to https://pypi.org/manage/account/publishing/
   - Add a new "pending publisher" with:
     - PyPI Project Name: `verskyt`
     - Owner: `jeffreyksmithjr`
     - Repository name: `verskyt`
     - Workflow filename: `publish.yml`
     - Environment name: (leave blank)

2. **Reserve Package Name** (optional but recommended):
   ```bash
   # Create a minimal package to reserve the name
   pip install build twine
   python -m build
   twine upload dist/* --repository testpypi  # Test first
   twine upload dist/*  # Then upload to real PyPI
   ```

### Publishing Releases

Publishing is **fully automated** via GitHub Actions:

```bash
# 1. Update version in pyproject.toml
# 2. Create and push a version tag
git tag v0.1.0
git push origin v0.1.0

# 3. GitHub Actions automatically:
#    - Builds wheel and source distributions
#    - Verifies the package can be imported
#    - Publishes to PyPI
#    - Creates a GitHub release
#    - Tests installation across Python versions
```

### Release Checklist

Before creating a release tag:

- [ ] Update version in `pyproject.toml`
- [ ] All tests pass (`pytest`)
- [ ] Documentation builds (`cd docs && make html`)
- [ ] CHANGELOG.md updated (if you create one)
- [ ] No uncommitted changes

### Monitoring

- **PyPI**: Check https://pypi.org/project/verskyt/
- **GitHub Releases**: Check https://github.com/jeffreyksmithjr/verskyt/releases
- **Downloads**: PyPI provides download statistics

## 🔍 Verification

### Documentation
- Local: `cd docs && make html && open _build/html/index.html`
- Read the Docs: https://verskyt.readthedocs.io/
- GitHub Pages: https://jeffreyksmithjr.github.io/verskyt/

### Package
- Test installation: `pip install verskyt`
- Import test: `python -c "from verskyt import TverskyProjectionLayer"`
- Version check: `python -c "import verskyt; print(verskyt.__version__)"`

## 🚨 Troubleshooting

### Documentation Issues
- **Build fails**: Check `pip install -e ".[dev]"` installs all dependencies
- **Missing pages**: Ensure all `.md`/`.rst` files are in `toctree` directives
- **Broken links**: Run `make linkcheck` in docs directory

### Package Issues  
- **PyPI upload fails**: Check trusted publishing configuration
- **Import fails**: Verify `__init__.py` exports are correct
- **Version mismatch**: Ensure tag matches `pyproject.toml` version

## 📋 Status

- [x] PyPI publishing workflow configured
- [x] Version tag automation working
- [x] Cross-platform testing enabled
- [ ] Documentation hosting (needs Read the Docs setup)
- [ ] Badge updates in README.md
- [ ] Release notes automation