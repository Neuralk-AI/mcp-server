from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from seldon_mcp import neuralk_sdk


class _FakeResource:
    """Records .create() calls and returns a canned response."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _fake_client(resp):
    return SimpleNamespace(
        classifications=_FakeResource(resp),
        regressions=_FakeResource(resp),
        datasets=_FakeResource(resp),
    )


_ARRS = dict(X_train=np.zeros((3, 2), np.float32), y_train=np.array([0, 1, 0]),
             X_test=np.zeros((1, 2), np.float32))
_RESP = {"predictions": np.array([1]), "probabilities": None, "model": "seldon-small"}


class TestBuildClient:
    def test_default_url_leaves_host_implicit(self):
        # The SDK refuses uploads when a host is explicitly set, so the default
        # cloud URL must NOT be forwarded as a host.
        client = neuralk_sdk.build_client("nk_live_x", neuralk_sdk.DEFAULT_CLOUD_URL)
        assert client._user_provided_host is False

    def test_default_url_trailing_slash(self):
        client = neuralk_sdk.build_client("nk_live_x", neuralk_sdk.DEFAULT_CLOUD_URL + "/")
        assert client._user_provided_host is False

    def test_custom_url_forwarded_as_host(self):
        client = neuralk_sdk.build_client("nk_live_x", "https://on-prem.internal:8000")
        assert client._user_provided_host is True
        assert client.host == "https://on-prem.internal:8000"


class TestNormalize:
    def test_numpy_arrays_become_lists(self):
        resp = {
            "predictions": np.array([1, 0, 1]),
            "probabilities": np.array([[0.1, 0.9], [0.8, 0.2]]),
            "request_id": "r1",
            "model": "seldon-small",
            "credits_consumed": 3,
            "latency_ms": 42,
        }
        out = neuralk_sdk._normalize(resp)
        assert out["predictions"] == [1, 0, 1]
        assert out["probabilities"] == [[0.1, 0.9], [0.8, 0.2]]
        assert out["request_id"] == "r1"

    def test_missing_predictions_default_empty(self):
        out = neuralk_sdk._normalize({})
        assert out["predictions"] == []
        assert out["probabilities"] is None

    def test_list_predictions_pass_through(self):
        out = neuralk_sdk._normalize({"predictions": [2, 2], "probabilities": None})
        assert out["predictions"] == [2, 2]
        assert out["probabilities"] is None


class TestAllowedTtl:
    def test_tiers(self):
        assert neuralk_sdk.ALLOWED_TTL_DAYS == (1, 7, 30, 90)


class TestInferenceHelpers:
    def test_predict_inline_routes_classification(self):
        client = _fake_client(_RESP)
        neuralk_sdk.predict_inline(
            client, **_ARRS, model="seldon-small", problem_type="classification", dataset_name="d",
        )
        assert len(client.classifications.calls) == 1
        assert client.regressions.calls == []

    def test_predict_inline_routes_regression(self):
        client = _fake_client(_RESP)
        neuralk_sdk.predict_inline(
            client, **_ARRS, model="seldon-small", problem_type="regression", dataset_name="d",
        )
        assert len(client.regressions.calls) == 1
        assert client.classifications.calls == []

    def test_predict_inline_normalizes_response(self):
        client = _fake_client(_RESP)
        out = neuralk_sdk.predict_inline(
            client, **_ARRS, model="seldon-small", problem_type="classification", dataset_name="d",
        )
        assert out["predictions"] == [1]  # numpy -> list

    def test_predict_by_reference_uses_classifications_and_key(self):
        client = _fake_client(_RESP)
        neuralk_sdk.predict_by_reference(client, "ds-key")
        assert client.classifications.calls == [{"dataset_key": "ds-key"}]
        assert client.regressions.calls == []

    def test_upload_dataset_passes_ttl(self):
        client = _fake_client({"dataset_id": "ds1", "ttl_days": 7})
        out = neuralk_sdk.upload_dataset(
            client, **_ARRS, model="seldon-small", problem_type="classification",
            dataset_name="d", ttl_days=7,
        )
        assert out["dataset_id"] == "ds1"
        assert client.datasets.calls[0]["ttl_days"] == 7
