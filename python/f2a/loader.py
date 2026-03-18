"""Python-side data loader — handles formats that the Rust core cannot.

The Rust core (Polars) natively supports: CSV, TSV, JSON, JSONL, Parquet, Feather.
For all other formats (Excel, HuggingFace, databases, statistical packages, URLs, etc.)
this module loads the data via pandas and converts it to a temp Parquet file
that Rust can consume.

Supported formats:
    - **Delimited text**: CSV, TSV, TXT (auto-detect), DAT, TAB, FWF
    - **JSON family**: JSON, JSONL, NDJSON
    - **Spreadsheets**: XLSX, XLS, XLSM, XLSB, ODS
    - **Binary/columnar**: Parquet, Feather, Arrow IPC, ORC, HDF5, Pickle
    - **Statistical packages**: SAS (.sas7bdat, .xpt), Stata (.dta), SPSS (.sav, .zsav, .por)
    - **Databases**: SQLite, DuckDB
    - **Markup**: XML, HTML (tables)
    - **Remote**: HTTP/HTTPS URL (auto-routing by extension)
    - **Platforms**: HuggingFace Datasets (hf://...)
"""

from __future__ import annotations

import csv
import logging
import re
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger("f2a.loader")

# ── Formats that Rust/Polars can handle directly ──────────────────────
_RUST_NATIVE_FORMATS = {"csv", "tsv", "json", "jsonl", "parquet", "feather"}

# ── HuggingFace helpers ──────────────────────────────────────────────
HF_PREFIXES = ("hf://", "huggingface://")
HF_URL_PATTERN = re.compile(
    r"^https?://huggingface\.co/datasets/"
    r"(?P<dataset>[^/?#]+(?:/[^/?#]+)?)"
    r"(?:/viewer(?:/(?P<config>[^/?#]+))?(?:/(?P<split>[^/?#]+))?)?",
    re.IGNORECASE,
)
URL_PREFIXES = ("http://", "https://", "ftp://")

# ── Extension → source-type mapping ─────────────────────────────────
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".csv": "csv", ".tsv": "tsv", ".txt": "delimited", ".dat": "delimited", ".tab": "tsv",
    ".json": "json", ".jsonl": "jsonl", ".ndjson": "jsonl",
    ".xlsx": "excel", ".xls": "excel", ".xlsm": "excel", ".xlsb": "excel", ".ods": "ods",
    ".parquet": "parquet", ".pq": "parquet",
    ".feather": "feather", ".ftr": "feather",
    ".arrow": "arrow_ipc", ".ipc": "arrow_ipc",
    ".orc": "orc",
    ".hdf": "hdf5", ".hdf5": "hdf5", ".h5": "hdf5",
    ".pkl": "pickle", ".pickle": "pickle",
    ".sas7bdat": "sas", ".xpt": "sas_xport",
    ".dta": "stata", ".sav": "spss", ".zsav": "spss", ".por": "spss",
    ".db": "sqlite", ".sqlite": "sqlite", ".sqlite3": "sqlite",
    ".ddb": "duckdb", ".duckdb": "duckdb",
    ".xml": "xml", ".html": "html", ".htm": "html",
    ".fwf": "fwf",
}

# ── Registry ─────────────────────────────────────────────────────────
_LOADER_REGISTRY: dict[str, str] = {
    "csv": "_load_csv", "tsv": "_load_tsv", "delimited": "_load_delimited", "fwf": "_load_fwf",
    "json": "_load_json", "jsonl": "_load_jsonl",
    "excel": "_load_excel", "ods": "_load_ods",
    "parquet": "_load_parquet", "feather": "_load_feather",
    "arrow_ipc": "_load_arrow_ipc", "orc": "_load_orc",
    "hdf5": "_load_hdf5", "pickle": "_load_pickle",
    "sas": "_load_sas", "sas_xport": "_load_sas_xport", "stata": "_load_stata", "spss": "_load_spss",
    "sqlite": "_load_sqlite", "duckdb": "_load_duckdb",
    "xml": "_load_xml", "html": "_load_html",
    "url_auto": "_load_url_auto",
    "hf": "_load_huggingface",
}


# =====================================================================
#  Public API
# =====================================================================

def resolve_source(source: str, **kwargs: Any) -> tuple[str, bool]:
    """Resolve a data source to a path that the Rust core can consume.

    Returns:
        (path, is_temp): If is_temp=True, the caller should delete the file
        after use.
    """
    source_type = detect_source_type(source)
    logger.info("Source type detected: %s → %s", source, source_type)

    if source_type in _RUST_NATIVE_FORMATS and not _is_remote(source):
        return source, False

    # Need Python-side loading → pandas → temp parquet
    import pandas as pd

    method_name = _LOADER_REGISTRY.get(source_type)
    if method_name is None:
        raise ValueError(f"Unsupported format: {source} (detected: {source_type})")

    loader_fn = globals().get(method_name)
    if loader_fn is None:
        raise ValueError(f"Loader not implemented: {method_name}")

    df = loader_fn(source, **kwargs)
    if df is None or df.empty:
        raise ValueError(f"Loaded empty DataFrame from: {source}")

    logger.info("Loaded via Python: %d rows × %d cols → converting to temp parquet", len(df), len(df.columns))
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp_path = tmp.name
    tmp.close()
    df.to_parquet(tmp_path, index=False)
    return tmp_path, True


def supported_formats() -> list[str]:
    """Return list of supported source types."""
    return sorted(_LOADER_REGISTRY.keys())


# =====================================================================
#  Detection
# =====================================================================

def detect_source_type(source: str) -> str:
    """Detect data source type from a source string."""
    if HF_URL_PATTERN.match(source):
        return "hf"
    for prefix in URL_PREFIXES:
        if source.lower().startswith(prefix):
            return _detect_url_type(source)
    for prefix in HF_PREFIXES:
        if source.startswith(prefix):
            return "hf"
    # org/dataset pattern
    if "/" in source and not Path(source).suffix:
        parts = source.split("/")
        if len(parts) == 2 and all(re.match(r"^[a-zA-Z0-9_-]+$", p) for p in parts):
            return "hf"
    # File extension
    path = Path(source)
    ext = path.suffix.lower()
    full_suffixes = "".join(path.suffixes).lower()
    if full_suffixes in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[full_suffixes]
    if ext in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[ext]
    # Content sniffing
    if path.exists() and path.is_file():
        sniffed = _sniff_content(path)
        if sniffed:
            return sniffed
    raise ValueError(f"Unsupported format: {source} (ext: {ext or 'none'})")


def _is_remote(source: str) -> bool:
    return any(source.lower().startswith(p) for p in URL_PREFIXES)


def _detect_url_type(url: str) -> str:
    from urllib.parse import urlparse
    ext = Path(urlparse(url).path).suffix.lower()
    return SUPPORTED_EXTENSIONS.get(ext, "url_auto")


def _sniff_content(path: Path, peek_bytes: int = 8192) -> str | None:
    try:
        with open(path, "rb") as f:
            header = f.read(peek_bytes)
    except (OSError, PermissionError):
        return None
    if header[:4] == b"PAR1":
        return "parquet"
    if header[:6] == b"ARROW1":
        return "arrow_ipc"
    if header[:3] == b"ORC":
        return "orc"
    if header[:8] == b"\x89HDF\r\n\x1a\n":
        return "hdf5"
    if header[:4] == b"FEA1":
        return "feather"
    if header[:16] == b"SQLite format 3\x00":
        return "sqlite"
    if header[:2] in (b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"):
        return "pickle"
    if header[:4] == b"PK\x03\x04":
        if b"xl/" in header or b"[Content_Types].xml" in header:
            return "excel"
        return None
    if header[:4] == b"\xd0\xcf\x11\xe0":
        return "excel"
    try:
        text = header.decode("utf-8", errors="replace").strip()
    except Exception:
        return None
    if text.startswith(("{", "[")):
        lines = text.split("\n", 5)
        if len(lines) > 1 and all(l.strip().startswith("{") for l in lines[:3] if l.strip()):
            return "jsonl"
        return "json"
    if text.startswith("<?xml") or text.startswith("<"):
        if "<html" in text.lower() or "<table" in text.lower():
            return "html"
        return "xml"
    if "\t" in text and text.count("\t") > text.count(","):
        return "tsv"
    if "," in text:
        return "csv"
    if "\n" in text and len(text.split("\n")) > 1:
        return "delimited"
    return None


# =====================================================================
#  Loaders  (each returns pd.DataFrame)
# =====================================================================

def _load_csv(source: str, **kw: Any):
    import pandas as pd
    kw.setdefault("encoding", "utf-8")
    try:
        return pd.read_csv(source, **kw)
    except UnicodeDecodeError:
        kw["encoding"] = "cp949"
        return pd.read_csv(source, **kw)


def _load_tsv(source: str, **kw: Any):
    import pandas as pd
    kw.setdefault("sep", "\t")
    return pd.read_csv(source, **kw)


def _load_delimited(source: str, **kw: Any):
    import pandas as pd
    if "sep" in kw or "delimiter" in kw:
        return pd.read_csv(source, **kw)
    try:
        with open(source, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(8192)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;: ")
        kw["sep"] = dialect.delimiter
        return pd.read_csv(source, **kw)
    except csv.Error:
        pass
    for sep in [",", "\t", ";", "|", " "]:
        try:
            df = pd.read_csv(source, sep=sep, nrows=5, **kw)
            if len(df.columns) > 1:
                return pd.read_csv(source, sep=sep, **kw)
        except Exception:
            continue
    return pd.read_csv(source, **kw)


def _load_fwf(source: str, **kw: Any):
    import pandas as pd
    return pd.read_fwf(source, **kw)


def _load_json(source: str, **kw: Any):
    import json as json_mod
    import pandas as pd
    try:
        return pd.read_json(source, **kw)
    except ValueError:
        with open(source, "r", encoding="utf-8") as f:
            data = json_mod.load(f)
        if isinstance(data, list):
            return pd.json_normalize(data)
        elif isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                    return pd.json_normalize(val)
            return pd.json_normalize(data)
        raise


def _load_jsonl(source: str, **kw: Any):
    import pandas as pd
    kw.setdefault("lines", True)
    return pd.read_json(source, **kw)


def _load_excel(source: str, **kw: Any):
    import pandas as pd
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise ImportError("Install 'openpyxl' for Excel support: pip install f2a[io]") from exc
    if Path(source).suffix.lower() == ".xlsb":
        try:
            import pyxlsb  # noqa: F401
            kw.setdefault("engine", "pyxlsb")
        except ImportError as exc:
            raise ImportError("Install 'pyxlsb' for xlsb support") from exc
    result = pd.read_excel(source, **kw)
    if isinstance(result, dict):
        sheet_names = list(result.keys())
        logger.warning("%d sheets found, using first: '%s'", len(sheet_names), sheet_names[0])
        return result[sheet_names[0]]
    return result


def _load_ods(source: str, **kw: Any):
    import pandas as pd
    try:
        import odf  # noqa: F401
    except ImportError as exc:
        raise ImportError("Install 'odfpy' for ODS support: pip install odfpy") from exc
    kw.setdefault("engine", "odf")
    return pd.read_excel(source, **kw)


def _load_parquet(source: str, **kw: Any):
    import pandas as pd
    return pd.read_parquet(source, **kw)


def _load_feather(source: str, **kw: Any):
    import pandas as pd
    return pd.read_feather(source, **kw)


def _load_arrow_ipc(source: str, **kw: Any):
    import pyarrow.ipc as ipc
    with open(source, "rb") as f:
        reader = ipc.open_file(f)
        table = reader.read_all()
    return table.to_pandas(**kw)


def _load_orc(source: str, **kw: Any):
    import pandas as pd
    return pd.read_orc(source, **kw)


def _load_hdf5(source: str, **kw: Any):
    import pandas as pd
    try:
        import tables  # noqa: F401
    except ImportError as exc:
        raise ImportError("Install 'tables' for HDF5 support") from exc
    key = kw.pop("key", None)
    if key:
        return pd.read_hdf(source, key=key, **kw)
    with pd.HDFStore(source, mode="r") as store:
        keys = store.keys()
        if not keys:
            raise ValueError("No datasets found in HDF5 file.")
        if len(keys) > 1:
            logger.warning("HDF5 has %d keys, using first: '%s'", len(keys), keys[0])
        return pd.read_hdf(source, key=keys[0], **kw)


def _load_pickle(source: str, **kw: Any):
    import pandas as pd
    logger.warning("Loading pickle — verify this is a trusted source: %s", source)
    return pd.read_pickle(source, **kw)


def _load_sas(source: str, **kw: Any):
    import pandas as pd
    kw.setdefault("format", "sas7bdat")
    return pd.read_sas(source, **kw)


def _load_sas_xport(source: str, **kw: Any):
    import pandas as pd
    kw.setdefault("format", "xport")
    return pd.read_sas(source, **kw)


def _load_stata(source: str, **kw: Any):
    import pandas as pd
    return pd.read_stata(source, **kw)


def _load_spss(source: str, **kw: Any):
    import pandas as pd
    try:
        import pyreadstat  # noqa: F401
    except ImportError as exc:
        raise ImportError("Install 'pyreadstat' for SPSS support: pip install pyreadstat") from exc
    return pd.read_spss(source, **kw)


def _load_sqlite(source: str, **kw: Any):
    import sqlite3
    import pandas as pd
    table = kw.pop("table", None)
    query = kw.pop("query", None)
    conn = sqlite3.connect(source)
    try:
        if query:
            return pd.read_sql_query(query, conn, **kw)
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )["name"].tolist()
        if not tables:
            raise ValueError("No tables found in SQLite database.")
        if table is None:
            table = tables[0]
            if len(tables) > 1:
                logger.warning("SQLite has %d tables, using '%s'", len(tables), table)
        if table not in tables:
            raise ValueError(f"Table '{table}' not found. Available: {tables}")
        return pd.read_sql_query(f'SELECT * FROM "{table}"', conn, **kw)
    finally:
        conn.close()


def _load_duckdb(source: str, **kw: Any):
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError("Install 'duckdb' for DuckDB support: pip install duckdb") from exc
    import pandas as pd
    table = kw.pop("table", None)
    query = kw.pop("query", None)
    conn = duckdb.connect(source, read_only=True)
    try:
        if query:
            return conn.execute(query).fetchdf()
        tables = conn.execute("SHOW TABLES").fetchdf()
        table_names = tables.iloc[:, 0].tolist() if not tables.empty else []
        if not table_names:
            raise ValueError("No tables found in DuckDB database.")
        if table is None:
            table = table_names[0]
            if len(table_names) > 1:
                logger.warning("DuckDB has %d tables, using '%s'", len(table_names), table)
        return conn.execute(f'SELECT * FROM "{table}"').fetchdf()
    finally:
        conn.close()


def _load_xml(source: str, **kw: Any):
    import pandas as pd
    try:
        import lxml  # noqa: F401
    except ImportError:
        kw.setdefault("parser", "etree")
    return pd.read_xml(source, **kw)


def _load_html(source: str, **kw: Any):
    import pandas as pd
    try:
        import lxml  # noqa: F401
    except ImportError:
        kw.setdefault("flavor", "bs4")
    table_index = kw.pop("table_index", None)
    tables = pd.read_html(source, **kw)
    if not tables:
        raise ValueError("No tables found in HTML file.")
    if table_index is not None:
        return tables[table_index]
    if len(tables) > 1:
        sizes = [(i, len(t) * len(t.columns)) for i, t in enumerate(tables)]
        best_idx = max(sizes, key=lambda x: x[1])[0]
        logger.warning("Found %d tables in HTML, using largest (#%d)", len(tables), best_idx)
        return tables[best_idx]
    return tables[0]


def _load_url_auto(source: str, **kw: Any):
    """Download and load a file from a URL."""
    from urllib.parse import urlparse
    from urllib.request import urlopen, Request

    logger.info("Downloading from URL: %s", source)
    req = Request(source, headers={"User-Agent": "f2a/1.0"})
    with urlopen(req, timeout=60) as resp:
        content_type = resp.headers.get("Content-Type", "").lower()
        data = resp.read()

    ct_map = {
        "text/csv": "csv", "text/tab-separated-values": "tsv",
        "application/json": "json", "application/x-ndjson": "jsonl",
        "application/vnd.apache.parquet": "parquet",
        "application/vnd.openxmlformats": "excel", "application/vnd.ms-excel": "excel",
        "text/xml": "xml", "application/xml": "xml", "text/html": "html",
    }
    detected_type = None
    for ct_key, fmt in ct_map.items():
        if ct_key in content_type:
            detected_type = fmt
            break
    if detected_type is None:
        path_ext = Path(urlparse(source).path).suffix.lower()
        detected_type = SUPPORTED_EXTENSIONS.get(path_ext, "csv")

    suffix_map = {
        "csv": ".csv", "tsv": ".tsv", "json": ".json", "jsonl": ".jsonl",
        "parquet": ".parquet", "excel": ".xlsx", "xml": ".xml", "html": ".html",
    }
    suffix = suffix_map.get(detected_type, ".tmp")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    method_name = _LOADER_REGISTRY.get(detected_type)
    if method_name and method_name in globals():
        try:
            return globals()[method_name](tmp_path, **kw)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    import pandas as pd
    try:
        return pd.read_csv(tmp_path, **kw)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _load_huggingface(source: str, **kw: Any):
    """Load a HuggingFace dataset.

    When neither config nor split is specified, all available
    configs × splits are discovered and concatenated.
    """
    import pandas as pd
    try:
        from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install 'datasets' for HuggingFace support: pip install f2a[io] or pip install datasets"
        ) from exc

    dataset_name = source
    hf_match = HF_URL_PATTERN.match(dataset_name)
    if hf_match:
        dataset_name = hf_match.group("dataset")
        url_config = hf_match.group("config")
        url_split = hf_match.group("split")
        if url_config and "config" not in kw:
            kw["config"] = url_config
        if url_split and "split" not in kw:
            kw["split"] = url_split
    else:
        for prefix in HF_PREFIXES:
            if dataset_name.startswith(prefix):
                dataset_name = dataset_name[len(prefix):]
                break

    dataset_name = dataset_name.rstrip("/")
    config = kw.pop("config", None)
    split = kw.pop("split", None)

    # Explicit single partition
    if config is not None or split is not None:
        split = split or "train"
        if config:
            ds = load_dataset(dataset_name, config, split=split, **kw)
        else:
            ds = load_dataset(dataset_name, split=split, **kw)
        return ds.to_pandas()

    # Auto-discover all configs × splits
    try:
        configs = get_dataset_config_names(dataset_name)
    except Exception:
        configs = [None]
    if not configs:
        configs = [None]

    frames: list = []
    for cfg in configs:
        try:
            splits = get_dataset_split_names(dataset_name, cfg) if cfg else get_dataset_split_names(dataset_name)
        except Exception:
            splits = ["train"]
        for sp in splits:
            try:
                ds = load_dataset(dataset_name, cfg, split=sp, **kw) if cfg else load_dataset(dataset_name, split=sp, **kw)
                df_part = ds.to_pandas()
                df_part["__subset__"] = cfg or "default"
                df_part["__split__"] = sp
                frames.append(df_part)
                logger.info("HF loaded: config=%s split=%s (%d rows)", cfg or "default", sp, len(df_part))
            except Exception as exc:
                logger.warning("Failed: config=%s split=%s: %s", cfg, sp, exc)

    if not frames:
        raise ValueError(f"No loadable configs/splits for: {source}")
    return pd.concat(frames, ignore_index=True)
