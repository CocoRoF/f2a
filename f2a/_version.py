"""Version information for f2a."""

try:
    from importlib.metadata import version as _get_version

    __version__: str = _get_version("f2a")
except Exception:
    __version__ = "1.1.1"
