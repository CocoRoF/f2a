"""Core module — data loading, analysis orchestration, and schema inference."""

from f2a.core.loader import DataLoader
from f2a.core.analyzer import analyze, Analyzer
from f2a.core.schema import DataSchema, infer_schema

__all__ = ["DataLoader", "analyze", "Analyzer", "DataSchema", "infer_schema"]
