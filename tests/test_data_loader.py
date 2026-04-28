from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from seldon_mcp.data_loader import (
    describe_dataframe,
    load_dataframe,
    resolve_path,
)

# --- resolve_path ---


class TestResolvePath:
    def test_relative_within_base(self, tmp_data_dir: Path):
        result = resolve_path("data.csv", str(tmp_data_dir))
        assert result == tmp_data_dir / "data.csv"

    def test_absolute_within_base(self, tmp_data_dir: Path):
        abs_path = str(tmp_data_dir / "data.csv")
        result = resolve_path(abs_path, str(tmp_data_dir))
        assert result == tmp_data_dir / "data.csv"

    def test_traversal_blocked(self, tmp_data_dir: Path):
        with pytest.raises(ValueError, match="Access denied"):
            resolve_path("../../etc/passwd", str(tmp_data_dir))

    def test_absolute_outside_blocked(self, tmp_data_dir: Path):
        with pytest.raises(ValueError, match="Access denied"):
            resolve_path("/etc/passwd", str(tmp_data_dir))

    def test_dot_dot_in_middle_blocked(self, tmp_data_dir: Path):
        with pytest.raises(ValueError, match="Access denied"):
            resolve_path("subdir/../../etc/passwd", str(tmp_data_dir))

    def test_subdirectory_allowed(self, tmp_data_dir: Path):
        subdir = tmp_data_dir / "sub"
        subdir.mkdir()
        result = resolve_path("sub/data.csv", str(tmp_data_dir))
        assert result == subdir / "data.csv"


# --- load_dataframe ---


class TestLoadDataframe:
    def test_load_csv(self, classification_csv: Path, tmp_data_dir: Path):
        df = load_dataframe(str(classification_csv), str(tmp_data_dir))
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (10, 3)
        assert "feature_a" in df.columns
        assert "label" in df.columns

    def test_load_parquet(self, parquet_file: Path, tmp_data_dir: Path):
        df = load_dataframe(str(parquet_file), str(tmp_data_dir))
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (10, 3)

    def test_load_json(self, json_file: Path, tmp_data_dir: Path):
        df = load_dataframe(str(json_file), str(tmp_data_dir))
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (3, 2)

    def test_file_not_found(self, tmp_data_dir: Path):
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_dataframe("nonexistent.csv", str(tmp_data_dir))

    def test_unsupported_format(self, tmp_data_dir: Path):
        bad_file = tmp_data_dir / "data.txt"
        bad_file.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_dataframe(str(bad_file), str(tmp_data_dir))

    def test_path_traversal_blocked(self, tmp_data_dir: Path):
        with pytest.raises(ValueError, match="Access denied"):
            load_dataframe("../../etc/passwd", str(tmp_data_dir))


# --- describe_dataframe ---


class TestDescribeDataframe:
    def test_basic_describe(self, classification_csv: Path, tmp_data_dir: Path):
        df = load_dataframe(str(classification_csv), str(tmp_data_dir))
        summary = describe_dataframe(df)

        assert summary["shape"] == {"rows": 10, "columns": 3}
        assert len(summary["columns"]) == 3

        col_names = [c["name"] for c in summary["columns"]]
        assert "feature_a" in col_names
        assert "feature_b" in col_names
        assert "label" in col_names

    def test_numeric_summary(self, classification_csv: Path, tmp_data_dir: Path):
        df = load_dataframe(str(classification_csv), str(tmp_data_dir))
        summary = describe_dataframe(df)

        numeric_cols = [s["column"] for s in summary["numeric_summary"]]
        assert "feature_a" in numeric_cols
        assert "feature_b" in numeric_cols
        assert "label" not in numeric_cols

        feature_a = next(s for s in summary["numeric_summary"] if s["column"] == "feature_a")
        assert feature_a["mean"] == 5.5
        assert feature_a["min"] == 1.0
        assert feature_a["max"] == 10.0

    def test_sample_rows_included(self, classification_csv: Path, tmp_data_dir: Path):
        df = load_dataframe(str(classification_csv), str(tmp_data_dir))
        summary = describe_dataframe(df, include_sample=True)
        assert "sample_rows" in summary
        assert len(summary["sample_rows"]) == 5

    def test_sample_rows_excluded(self, classification_csv: Path, tmp_data_dir: Path):
        df = load_dataframe(str(classification_csv), str(tmp_data_dir))
        summary = describe_dataframe(df, include_sample=False)
        assert "sample_rows" not in summary

    def test_null_counts(self, nulls_csv: Path, tmp_data_dir: Path):
        df = load_dataframe(str(nulls_csv), str(tmp_data_dir))
        summary = describe_dataframe(df)

        x_col = next(c for c in summary["columns"] if c["name"] == "x")
        assert x_col["null_count"] == 2
        assert x_col["null_pct"] == 40.0

        y_col = next(c for c in summary["columns"] if c["name"] == "y")
        assert y_col["null_count"] == 5
        assert y_col["null_pct"] == 100.0

    def test_all_null_numeric_column(self, nulls_csv: Path, tmp_data_dir: Path):
        """Column 'y' is all nulls — should not crash."""
        df = load_dataframe(str(nulls_csv), str(tmp_data_dir))
        summary = describe_dataframe(df)

        numeric_cols = [s["column"] for s in summary["numeric_summary"]]
        # 'y' has no non-null values, so it should not appear in numeric_summary
        assert "y" not in numeric_cols
        # 'x' should still be there
        assert "x" in numeric_cols

    def test_empty_dataframe(self, empty_csv: Path, tmp_data_dir: Path):
        df = load_dataframe(str(empty_csv), str(tmp_data_dir))
        summary = describe_dataframe(df)
        assert summary["shape"]["rows"] == 0
        assert "sample_rows" not in summary
