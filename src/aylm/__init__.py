"""AYLM - Advanced YLM for 3D Gaussian Splatting."""

__version__ = "2.0.0"

__all__ = ["__version__"]


def _lazy_import(name):
    """Lazy import to avoid circular dependencies."""
    import importlib

    return importlib.import_module(f".{name}", __name__)
