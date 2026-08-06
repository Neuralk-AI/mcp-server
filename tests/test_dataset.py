from __future__ import annotations

import numpy as np
import polars as pl

from seldon_mcp.dataset import (
    decode_predictions,
    detect_problem_type,
    parse_csv,
    prepare_arrays,
    prepare_data,
    prepare_target,
)


class TestParseCsv:
    def test_parses_inline_text(self):
        df = parse_csv("a,b\n1,x\n2,y\n")
        assert df.columns == ["a", "b"]
        assert df.height == 2


class TestDetectProblemType:
    def test_continuous_is_regression(self):
        assert detect_problem_type(np.array([1.5, 2.7, 3.1, 9.9])) == "regression"

    def test_discrete_is_classification(self):
        assert detect_problem_type(np.array([0, 1, 1, 0])) == "classification"

    def test_strings_are_classification(self):
        assert detect_problem_type(np.array(["a", "b", "a"])) == "classification"


class TestPrepareTarget:
    def test_regression_casts_float64(self):
        y, classes = prepare_target(np.array([100, 200, 300]), "regression")
        assert y.dtype == np.float64
        assert classes is None

    def test_classification_string_encoded_int64(self):
        y, classes = prepare_target(np.array(["cat", "dog", "cat"]), "classification")
        assert y.dtype == np.int64
        assert classes == ["cat", "dog"]
        assert y.tolist() == [0, 1, 0]

    def test_classification_integer_passthrough(self):
        y, classes = prepare_target(np.array([0, 1, 1]), "classification")
        assert y.dtype == np.int64
        assert classes is None

    def test_round_trip_decode(self):
        y, classes = prepare_target(np.array(["b", "a", "b", "c"]), "classification")
        assert decode_predictions(y.tolist(), classes) == ["b", "a", "b", "c"]


class TestDecodePredictions:
    def test_no_classes_passthrough(self):
        assert decode_predictions([1.5, 2.5], None) == [1.5, 2.5]

    def test_maps_codes_to_labels(self):
        assert decode_predictions([0, 1, 1], ["no", "yes"]) == ["no", "yes", "yes"]

    def test_out_of_range_left_as_is(self):
        assert decode_predictions([5], ["a", "b"]) == [5]


class TestPrepareData:
    def test_missing_target_raises(self):
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        try:
            prepare_data(df, "missing")
        except ValueError as e:
            assert "not found" in str(e)
        else:
            raise AssertionError("expected ValueError")

    def test_string_features_encoded_numeric(self):
        # pyarrow-backed string column must be detected and vectorized to numeric
        df = pl.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "cat": ["x", "y", "x", "y"], "y": [0, 1, 0, 1]})
        X, y, vec = prepare_data(df, "y")
        assert vec is not None
        assert np.asarray(X, dtype=np.float32).dtype == np.float32
        assert y.tolist() == [0, 1, 0, 1]

    def test_predict_set_without_target_allowed(self):
        # An unlabeled predict set (no target column) must be accepted when
        # require_target=False; y comes back as None and all columns are features.
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        X, y, _ = prepare_data(df, "y", require_target=False)
        assert y is None
        assert list(X.columns) == ["a", "b"]

    def test_predict_set_missing_target_still_raises_when_required(self):
        df = pl.DataFrame({"a": [1, 2]})
        try:
            prepare_data(df, "y", require_target=True)
        except ValueError as e:
            assert "not found" in str(e)
        else:
            raise AssertionError("expected ValueError")


class TestPrepareArrays:
    def test_holdout_split_string_target(self):
        rows = {"a": list(range(20)), "b": [float(i) for i in range(20)],
                "species": ["x" if i % 2 else "y" for i in range(20)]}
        df = pl.DataFrame(rows)
        X_train, y_train, X_test, classes, feats, pt = prepare_arrays(df, "species", holdout_size=0.25)
        assert X_train.dtype == np.float32
        assert X_test.dtype == np.float32
        assert pt == "classification"
        assert y_train.dtype == np.int64
        assert sorted(classes) == ["x", "y"]
        assert len(X_train) == 15 and len(X_test) == 5

    def test_continuous_target_is_regression(self):
        rows = {"a": [float(i) for i in range(20)], "target": [i * 1.7 + 0.5 for i in range(20)]}
        df = pl.DataFrame(rows)
        X_train, y_train, X_test, classes, feats, pt = prepare_arrays(df, "target", holdout_size=0.25)
        assert pt == "regression"
        assert y_train.dtype == np.float64
        assert classes is None

    def test_explicit_problem_type_overrides(self):
        # integer target would auto-detect as classification; force regression
        df = pl.DataFrame({"a": [float(i) for i in range(20)], "y": list(range(20))})
        _, y_train, _, classes, _, pt = prepare_arrays(df, "y", holdout_size=0.25, problem_type="regression")
        assert pt == "regression" and y_train.dtype == np.float64 and classes is None

    def test_separate_predict_df(self):
        ctx = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]})
        pred = pl.DataFrame({"a": [5.0, 6.0], "y": [0, 0]})
        X_train, y_train, X_test, classes, feats, pt = prepare_arrays(ctx, "y", predict_df=pred)
        assert len(X_train) == 4 and len(X_test) == 2
        assert classes is None  # numeric target

    def test_separate_predict_df_unlabeled(self):
        # The predict set carries NO target column — the documented flow. This used
        # to raise "Target column not found"; it must now predict on all rows.
        ctx = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "species": ["x", "y", "x", "y"]})
        pred = pl.DataFrame({"a": [5.0, 6.0, 7.0]})  # no "species" column
        X_train, y_train, X_test, classes, feats, pt = prepare_arrays(ctx, "species", predict_df=pred)
        assert len(X_train) == 4 and len(X_test) == 3
        assert pt == "classification"
        assert sorted(classes) == ["x", "y"]
