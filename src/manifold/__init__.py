"""Manifold SDK: the primitives a policy and a benchmark are paired through."""

from importlib.metadata import version

from manifold.recipes.local import serve

# Single-sourced from the installed package metadata (pyproject `version`).
__version__ = version("manifold-sdk")

__all__ = [
    "__version__",
    "serve",
]
