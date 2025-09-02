# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# Add the project root to the Python path so Sphinx can import the module
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "Verskyt"
copyright = "2025, Verskyt Contributors"
author = "Verskyt Contributors"
release = "0.2.4"
version = "0.2.4"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # Core library to pull in docstring documentation
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx.ext.intersphinx",  # Link to other project documentation
    # 'myst_parser',  # Parse Markdown files - commented due to compatibility issues
    # 'myst_nb',      # Parse Jupyter notebooks - commented due to compatibility issues
    # 'sphinx_autodoc_typehints',  # Better type hints - commented due to compatibility
]

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Notebook execution configuration
nb_execution_mode = "off"  # Don't execute notebooks during build

# Source file suffixes
source_suffix = {
    ".rst": None,
    # '.md': 'myst_parser',
    # '.ipynb': 'myst_nb',
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"

html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "logo-light.svg",
    "dark_logo": "logo-dark.svg",
    "source_repository": "https://github.com/verskyt/verskyt",
    "source_branch": "main",
    "source_directory": "docs/",
}

html_title = f"Verskyt {version}"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Don't show module names before class/function names
add_module_names = False

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Intersphinx mapping -----------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
