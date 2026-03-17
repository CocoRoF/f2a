from importlib.metadata import version, PackageNotFoundError

try:
    __version__: str = version("f2a")
except PackageNotFoundError:
    # Fallback for editable / dev installs where metadata isn't available yet
    __version__ = "0.0.0-dev"
