"""Tests: import, version, Rust core, AnalysisConfig."""

from __future__ import annotations

import json


def test_import():
    import f2a

    assert hasattr(f2a, "__version__")
    assert hasattr(f2a, "analyze")
    assert hasattr(f2a, "AnalysisConfig")


def test_version():
    import f2a

    # Version must be a valid semver-like string, not the dev fallback
    assert f2a.__version__ != "0.0.0-dev"
    parts = f2a.__version__.split(".")
    assert len(parts) >= 2, f"Unexpected version format: {f2a.__version__}"


def test_rust_core_version():
    from f2a._core import version
    import f2a

    # Rust core version (from Cargo.toml) must match Python package version (from pyproject.toml)
    assert version() == f2a.__version__, (
        f"Version mismatch: Rust core={version()}, Python package={f2a.__version__}"
    )


def test_rust_core_configs():
    from f2a._core import default_config, fast_config, minimal_config

    cfg = json.loads(default_config())
    assert cfg["descriptive"] is True
    assert cfg["correlation"] is True

    mcfg = json.loads(minimal_config())
    assert mcfg["correlation"] is False

    fcfg = json.loads(fast_config())
    assert fcfg["pca"] is False


def test_analysis_config_defaults():
    from f2a import AnalysisConfig

    cfg = AnalysisConfig()
    assert cfg.descriptive is True
    assert cfg.advanced is True


def test_analysis_config_presets():
    from f2a import AnalysisConfig

    assert AnalysisConfig.minimal().correlation is False
    assert AnalysisConfig.fast().pca is False
    assert AnalysisConfig.basic_only().advanced is False


def test_analysis_config_to_json():
    from f2a import AnalysisConfig

    parsed = json.loads(AnalysisConfig().to_json())
    assert isinstance(parsed, dict)
    assert "descriptive" in parsed
