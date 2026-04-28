from __future__ import annotations

from pathlib import Path

import polars as pl

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet", ".json"}


def resolve_path(file_path: str, base_dir: str = ".") -> Path:
    """Resolve a file path, ensuring it stays within base_dir.

    Raises ValueError if the resolved path escapes the base directory.
    """
    base = Path(base_dir).resolve()
    p = Path(file_path)
    if not p.is_absolute():
        p = base / p
    resolved = p.resolve()

    # Prevent path traversal outside the data directory
    try:
        resolved.relative_to(base)
    except ValueError:
        raise ValueError(
            f"Access denied: '{file_path}' resolves outside the data directory '{base}'"
        ) from None

    return resolved


def load_dataframe(file_path: str, base_dir: str = ".") -> pl.DataFrame:
    """Load a tabular data file into a polars DataFrame.

    Supports CSV, Excel (.xlsx/.xls), Parquet, and JSON files.
    """
    path = resolve_path(file_path, base_dir)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    if ext == ".csv":
        return pl.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        return pl.read_excel(path)
    elif ext == ".parquet":
        return pl.read_parquet(path)
    elif ext == ".json":
        return pl.read_json(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def describe_dataframe(df: pl.DataFrame, include_sample: bool = True) -> dict:
    """Generate a statistical summary of a polars DataFrame."""
    columns_info = []
    for col in df.columns:
        col_data = df[col]
        info = {
            "name": col,
            "dtype": str(col_data.dtype),
            "null_count": col_data.null_count(),
            "null_pct": round(col_data.null_count() / len(df) * 100, 2) if len(df) > 0 else 0,
            "unique_count": col_data.n_unique(),
        }
        columns_info.append(info)

    numeric_summary = []
    for col in df.columns:
        if df[col].dtype.is_numeric():
            col_data = df[col].drop_nulls()
            if len(col_data) > 0:
                mean = col_data.mean()
                median = col_data.median()
                std = col_data.std() if len(col_data) > 1 else None
                q25 = col_data.quantile(0.25)
                q75 = col_data.quantile(0.75)
                numeric_summary.append({
                    "column": col,
                    "mean": round(mean, 4) if mean is not None else None,
                    "std": round(std, 4) if std is not None else None,
                    "min": col_data.min(),
                    "max": col_data.max(),
                    "median": round(median, 4) if median is not None else None,
                    "q25": round(q25, 4) if q25 is not None else None,
                    "q75": round(q75, 4) if q75 is not None else None,
                })

    result = {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "columns": columns_info,
        "numeric_summary": numeric_summary,
    }

    if include_sample and len(df) > 0:
        sample = df.head(5)
        result["sample_rows"] = sample.to_dicts()

    return result
