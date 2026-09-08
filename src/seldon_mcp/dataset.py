"""Pure tabular-data preparation shared by the MCP server (inline tools) and the
upload CLI. This module performs NO file IO — callers pass already-loaded
polars DataFrames (or inline CSV text). It turns tabular data into the numeric
arrays the prediction archive needs.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import polars as pl
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target
from skrub import TableVectorizer


def parse_csv(data: str) -> pl.DataFrame:
    """Parse inline CSV text into a polars DataFrame (no file is touched)."""
    return pl.read_csv(io.BytesIO(data.encode("utf-8")))


def prepare_data(
    df: pl.DataFrame,
    target_column: str,
    feature_columns: list[str] | None = None,
    vectorizer: TableVectorizer | None = None,
    require_target: bool = True,
):
    """Extract features and target from a polars DataFrame, returning numeric data.

    If the feature matrix contains non-numeric columns, a TableVectorizer is fitted
    (or the provided one is used to transform) so the data is fully numeric.

    Set ``require_target=False`` for a separate predict set that carries no labels:
    the target column may be absent, in which case ``y`` is returned as None.

    Returns (X, y, vectorizer) — pass the fitted vectorizer when preparing a
    separate predict set so encoding is consistent.
    """
    pdf = df.to_pandas()
    target_present = target_column in pdf.columns
    if require_target and not target_present:
        available = ", ".join(pdf.columns.tolist())
        raise ValueError(f"Target column '{target_column}' not found. Available columns: {available}")

    if feature_columns:
        missing = [c for c in feature_columns if c not in pdf.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {', '.join(missing)}")
        X = pdf[feature_columns]
    elif target_present:
        X = pdf.drop(columns=[target_column])
    else:
        # Unlabeled predict set: no target column to drop, use all columns as features.
        X = pdf

    # to_numpy() yields a numpy-backed array even for pyarrow/string-backed columns,
    # which keeps the target indexable by train_test_split and encodable downstream.
    y = pdf[target_column].to_numpy() if target_present else None

    # Detect non-numeric via is_numeric_dtype so pyarrow-backed string/categorical
    # columns (dtype "string"/"str") are routed through the vectorizer.
    has_non_numeric = not all(is_numeric_dtype(X[col]) for col in X.columns)
    if has_non_numeric:
        if vectorizer is None:
            vectorizer = TableVectorizer()
            X = vectorizer.fit_transform(X)
        else:
            X = vectorizer.transform(X)

    return X, y, vectorizer


def detect_problem_type(y: np.ndarray) -> str:
    """Infer the task the way the Neuralk SDK does: discrete labels ->
    classification, continuous values -> regression."""
    return "classification" if type_of_target(np.asarray(y)) in ("binary", "multiclass") else "regression"


def prepare_target(y: np.ndarray, problem_type: str) -> tuple[np.ndarray, list[Any] | None]:
    """Cast/encode the target to match what the API expects per task type.

    - regression -> float64 (the dtype itself signals continuous output)
    - classification -> int64 (label-encoded first if the labels aren't integers)

    Returns (y_array, label_classes); label_classes is None unless string labels
    were encoded, in which case it's the ordered class list for decoding predictions.
    The API loads arrays with allow_pickle=False, so the result is always numeric.
    """
    arr = np.asarray(y)
    if problem_type == "regression":
        return arr.astype(np.float64), None
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64), None
    encoder = LabelEncoder()
    return encoder.fit_transform(arr).astype(np.int64), encoder.classes_.tolist()


def decode_predictions(values: list[Any], classes: list[Any] | None) -> list[Any]:
    """Map integer-coded predictions back to original labels when encoded."""
    if classes is None:
        return values
    decoded = []
    for v in values:
        try:
            decoded.append(classes[int(v)])
        except (ValueError, IndexError, TypeError):
            decoded.append(v)
    return decoded


def prepare_arrays(
    context_df: pl.DataFrame,
    target_column: str,
    predict_df: pl.DataFrame | None = None,
    feature_columns: list[str] | None = None,
    holdout_size: float = 0.2,
    random_state: int = 42,
    problem_type: str | None = None,
):
    """Turn a context DataFrame (and optional predict DataFrame) into the numeric
    arrays for the upload archive.

    Returns (X_train, y_train, X_test, label_classes, feature_names, problem_type).
    When predict_df is None, a holdout split of context_df is used as the predict
    set. problem_type is honored if given, else auto-detected from the target.
    """
    X_context, y_context, vec = prepare_data(context_df, target_column, feature_columns)
    problem_type = problem_type or detect_problem_type(y_context)

    if predict_df is not None:
        # The predict set is unlabeled: don't require the target column (its y is discarded).
        X_predict, _, _ = prepare_data(
            predict_df, target_column, feature_columns, vectorizer=vec, require_target=False,
        )
    else:
        X_context, X_predict, y_context, _ = train_test_split(
            X_context, y_context, test_size=holdout_size, random_state=random_state,
        )

    feature_names = list(X_context.columns)
    X_train = np.asarray(X_context, dtype=np.float32)
    X_test = np.asarray(X_predict, dtype=np.float32)
    y_train, label_classes = prepare_target(y_context, problem_type)
    return X_train, y_train, X_test, label_classes, feature_names, problem_type
