#!/usr/bin/env python
"""Setup script for Verskyt - backup for compatibility."""

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        packages=find_packages(),
        python_requires=">=3.8",
    )