"""Top-level entry-point for the map_fixations_on_face package"""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("pupil_labs.map_fixations_on_face")
except PackageNotFoundError:
    # package is not installed
    pass

__all__ = ["__version__"]
