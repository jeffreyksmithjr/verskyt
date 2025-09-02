"""
Visualization tools for Tversky Neural Networks.

This module provides visualization and interpretation tools for TNNs,
including prototype space visualization and data-based prototype interpretation.
"""

try:
    from verskyt.visualizations.plotting import (  # noqa: F401
        plot_prototype_space,
        visualize_prototypes_as_data,
    )

    __all__ = [
        "plot_prototype_space",
        "visualize_prototypes_as_data",
    ]

except ImportError as e:
    # Handle case where visualization dependencies are not installed
    import warnings

    warnings.warn(
        f"Visualization dependencies not available: {e}. "
        "Install with: pip install verskyt[visualization]",
        ImportWarning,
    )

    __all__ = []
