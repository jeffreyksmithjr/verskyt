#!/usr/bin/env python
"""Setup script for Verskyt - backup for compatibility."""

from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        packages=find_packages(),
        python_requires=">=3.8",
    )
