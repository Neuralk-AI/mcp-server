from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for test data files."""
    return tmp_path


@pytest.fixture
def classification_csv(tmp_data_dir: Path) -> Path:
    """Create a small classification CSV dataset."""
    df = pl.DataFrame({
        "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feature_b": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        "label": ["cat", "dog", "cat", "dog", "cat", "dog", "cat", "dog", "cat", "dog"],
    })
    path = tmp_data_dir / "classification.csv"
    df.write_csv(path)
    return path


@pytest.fixture
def regression_csv(tmp_data_dir: Path) -> Path:
    """Create a small regression CSV dataset."""
    df = pl.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "x2": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
        "target": [2.5, 5.5, 8.5, 11.5, 14.5, 17.5, 20.5, 23.5, 26.5, 29.5],
    })
    path = tmp_data_dir / "regression.csv"
    df.write_csv(path)
    return path


@pytest.fixture
def empty_csv(tmp_data_dir: Path) -> Path:
    """Create a CSV with headers but no rows."""
    path = tmp_data_dir / "empty.csv"
    path.write_text("a,b,c\n")
    return path


@pytest.fixture
def nulls_csv(tmp_data_dir: Path) -> Path:
    """Create a CSV with null values in numeric columns."""
    df = pl.DataFrame({
        "x": [1.0, None, 3.0, None, 5.0],
        "y": [None, None, None, None, None],
        "label": ["a", "b", "a", "b", "a"],
    })
    path = tmp_data_dir / "nulls.csv"
    df.write_csv(path)
    return path


@pytest.fixture
def parquet_file(tmp_data_dir: Path, classification_csv: Path) -> Path:
    """Create a Parquet version of the classification data."""
    df = pl.read_csv(classification_csv)
    path = tmp_data_dir / "data.parquet"
    df.write_parquet(path)
    return path


@pytest.fixture
def json_file(tmp_data_dir: Path) -> Path:
    """Create a JSON tabular file."""
    df = pl.DataFrame({
        "col1": [1, 2, 3],
        "col2": ["a", "b", "c"],
    })
    path = tmp_data_dir / "data.json"
    df.write_json(path)
    return path
