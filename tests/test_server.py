from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from neuralk.exceptions import NeuralkException

from seldon_mcp import neuralk_sdk, server
from seldon_mcp.auth import APIKeyAuthError, _SkipValidation
from seldon_mcp.config import SeldonConfig
from seldon_mcp.downloads import DownloadStore
from seldon_mcp.server import (
    SELDON_MODELS,
    VALID_MODEL_NAMES,
    VALID_TASK_TYPES,
    _predictions_csv,
    _resolve_api_key,
    _sanitize_error,
    _validate_params,
    complete_upload,
    create_upload,
    download_predictions,
    predict,
    predict_from_data,
    upload_data,
)


def _ctx(config: SeldonConfig, request=None):
    """Build a minimal Context stand-in. request=None is stdio-style (no HTTP)."""
    rc = SimpleNamespace(lifespan_context={"config": config}, request=request)
    return SimpleNamespace(request_context=rc)


def _http_request(headers=None, base_url="http://test/"):
    """A stub Starlette-style request with headers + base_url for HTTP-path tests."""
    return SimpleNamespace(headers=headers or {}, base_url=base_url)


_CLASSIFICATION_CSV = "a,b,label\n1,2,cat\n3,4,dog\n5,6,cat\n7,8,dog\n9,10,cat\n11,12,dog"

# --- _sanitize_error ---


class TestSanitizeError:
    def test_redacts_server_key(self):
        config = MagicMock()
        config.neuralk_api_key = "nk_live_secret123"
        err = Exception("Auth failed for nk_live_secret123")
        result = _sanitize_error(err, config)
        assert "nk_live_secret123" not in result
        assert "***" in result

    def test_redacts_request_key(self):
        config = MagicMock()
        config.neuralk_api_key = None
        err = Exception("Auth failed for nk_live_user_key")
        result = _sanitize_error(err, config, request_api_key="nk_live_user_key")
        assert "nk_live_user_key" not in result
        assert "***" in result

    def test_redacts_both_keys(self):
        config = MagicMock()
        config.neuralk_api_key = "server_key"
        err = Exception("server_key and user_key both failed")
        result = _sanitize_error(err, config, request_api_key="user_key")
        assert "server_key" not in result
        assert "user_key" not in result

    def test_no_key_no_change(self):
        config = MagicMock()
        config.neuralk_api_key = None
        err = Exception("Something went wrong")
        result = _sanitize_error(err, config)
        assert result == "Something went wrong"

    def test_redacts_key_in_prediction_api_error(self):
        from seldon_mcp.prediction_client import PredictionAPIError

        config = MagicMock()
        config.neuralk_api_key = "nk_live_secret123"
        err = PredictionAPIError(401, "bad key nk_live_secret123")
        result = _sanitize_error(err, config)
        assert "nk_live_secret123" not in result
        assert "***" in result


# --- prediction download (single-use) ---


class TestPredictionDownload:
    def test_csv_with_probabilities(self):
        out = _predictions_csv(["a", "b"], [[0.1, 0.9], [0.8, 0.2]]).decode()
        lines = out.strip().splitlines()
        assert lines[0] == "prediction,prob_0,prob_1"
        assert lines[1] == "a,0.1,0.9"

    def test_csv_without_probabilities(self):
        out = _predictions_csv([1.5, 2.5, 3.5], None).decode()
        lines = out.strip().splitlines()
        assert lines[0] == "prediction"
        assert lines[1:] == ["1.5", "2.5", "3.5"]

    def test_download_is_single_use(self, tmp_path, monkeypatch):
        store = DownloadStore(tmp_path)
        monkeypatch.setattr(server, "_download_store", store)
        token = store.save(b"prediction\n1\n")
        resp = asyncio.run(download_predictions(SimpleNamespace(path_params={"token": token})))
        assert resp.status_code == 200
        assert b"prediction" in resp.body
        # second fetch is gone (deleted on serve)
        resp2 = asyncio.run(download_predictions(SimpleNamespace(path_params={"token": token})))
        assert resp2.status_code == 404

    def test_unknown_token_404(self, tmp_path, monkeypatch):
        monkeypatch.setattr(server, "_download_store", DownloadStore(tmp_path))
        resp = asyncio.run(download_predictions(SimpleNamespace(path_params={"token": "nope"})))
        assert resp.status_code == 404


# --- _validate_params ---


class TestValidateParams:
    def test_valid_params(self):
        _validate_params(0.2, "seldon-small", 100, False, task_type="classification")

    def test_valid_with_separate_file(self):
        # With separate file, holdout_size and row count are irrelevant
        _validate_params(0.2, None, 1, True)

    def test_invalid_holdout_zero(self):
        with pytest.raises(ValueError, match="holdout_size must be between"):
            _validate_params(0.0, None, 100, False)

    def test_invalid_holdout_one(self):
        with pytest.raises(ValueError, match="holdout_size must be between"):
            _validate_params(1.0, None, 100, False)

    def test_invalid_holdout_negative(self):
        with pytest.raises(ValueError, match="holdout_size must be between"):
            _validate_params(-0.5, None, 100, False)

    def test_invalid_holdout_above_one(self):
        with pytest.raises(ValueError, match="holdout_size must be between"):
            _validate_params(1.5, None, 100, False)

    def test_invalid_model(self):
        with pytest.raises(ValueError, match="Unknown model"):
            _validate_params(0.2, "seldon-xxl", 100, False)

    def test_invalid_task_type(self):
        with pytest.raises(ValueError, match="Unknown task_type"):
            _validate_params(0.2, None, 100, False, task_type="clustering")

    def test_too_few_rows(self):
        with pytest.raises(ValueError, match="Need at least"):
            _validate_params(0.2, None, 2, False)

    def test_all_valid_models_accepted(self):
        for model_name in VALID_MODEL_NAMES:
            _validate_params(0.2, model_name, 100, False)

    def test_all_valid_task_types_accepted(self):
        for task in VALID_TASK_TYPES:
            _validate_params(0.2, None, 100, False, task_type=task)

    def test_none_task_type_accepted(self):
        _validate_params(0.2, None, 100, False, task_type=None)


# --- list_models tool ---


class TestListModels:
    def test_models_structure(self):
        for m in SELDON_MODELS:
            assert "name" in m
            assert "description" in m
            assert "tier" in m

    def test_three_variants(self):
        assert len(SELDON_MODELS) == 3
        names = {m["name"] for m in SELDON_MODELS}
        assert names == {"seldon-flash", "seldon-small", "seldon-large"}


# --- MCP server registration ---


class TestServerRegistration:
    def test_tools_registered(self):
        from seldon_mcp.server import mcp

        tool_names = {t.name for t in mcp._tool_manager.list_tools()}
        assert tool_names == {
            "predict", "predict_from_data", "upload_data",
            "create_upload", "complete_upload", "list_models",
        }

    def test_resources_registered(self):
        from seldon_mcp.server import mcp

        resource_uris = {str(r.uri) for r in mcp._resource_manager.list_resources()}
        assert "seldon://models" in resource_uris
        assert "seldon://config" in resource_uris

    def test_prompts_registered(self):
        from seldon_mcp.server import mcp

        prompt_names = {p.name for p in mcp._prompt_manager.list_prompts()}
        assert prompt_names == {"drop_and_predict"}

    def test_server_has_instructions(self):
        from seldon_mcp.server import mcp

        assert mcp.instructions is not None
        assert "in-context learning" in mcp.instructions
        assert "context selection" in mcp.instructions.upper() or "context selection" in mcp.instructions


# --- SDK-backed tools (SDK mocked) ---


class TestUploadData:
    async def test_uploads_and_returns_label_classes(self, monkeypatch):
        captured = {}

        def fake_upload(client, **kwargs):
            captured.update(kwargs)
            return {"dataset_id": "ds1", "bytes": 42, "etag": "e1", "ttl_days": kwargs.get("ttl_days")}

        monkeypatch.setattr(neuralk_sdk, "build_client", lambda *a, **k: "CLIENT")
        monkeypatch.setattr(neuralk_sdk, "upload_dataset", fake_upload)

        config = SeldonConfig(neuralk_api_key="nk_test", skip_api_key_validation=True)
        out = json.loads(await upload_data(_ctx(config), data=_CLASSIFICATION_CSV, target_column="label", ttl_days=7))

        assert out["dataset_id"] == "ds1"
        assert out["ttl_days"] == 7
        # string labels -> label_classes returned for decoding later
        assert out["label_classes"] == ["cat", "dog"]
        assert captured["ttl_days"] == 7
        assert captured["problem_type"] == "classification"

    async def test_config_ttl_default_applied(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(neuralk_sdk, "build_client", lambda *a, **k: "CLIENT")
        monkeypatch.setattr(
            neuralk_sdk, "upload_dataset",
            lambda client, **k: captured.update(k) or {"dataset_id": "ds1"},
        )
        config = SeldonConfig(neuralk_api_key="nk_test", seldon_upload_ttl_days=30, skip_api_key_validation=True)
        await upload_data(_ctx(config), data=_CLASSIFICATION_CSV, target_column="label")
        assert captured["ttl_days"] == 30

    async def test_invalid_ttl_rejected(self, monkeypatch):
        monkeypatch.setattr(neuralk_sdk, "build_client", lambda *a, **k: "CLIENT")
        config = SeldonConfig(neuralk_api_key="nk_test", skip_api_key_validation=True)
        out = json.loads(await upload_data(_ctx(config), data=_CLASSIFICATION_CSV, target_column="label", ttl_days=5))
        assert "Invalid ttl_days" in out["error"]


class TestPredictFromData:
    async def test_decodes_string_labels(self, monkeypatch):
        monkeypatch.setattr(neuralk_sdk, "build_client", lambda *a, **k: "CLIENT")
        monkeypatch.setattr(
            neuralk_sdk, "predict_inline",
            lambda client, **k: {"predictions": [0, 1], "probabilities": [[0.9, 0.1], [0.2, 0.8]],
                                 "model": "seldon-small"},
        )
        config = SeldonConfig(neuralk_api_key="nk_test", skip_api_key_validation=True)
        out = json.loads(await predict_from_data(_ctx(config), data=_CLASSIFICATION_CSV, target_column="label"))
        assert out["predictions"] == ["cat", "dog"]
        assert out["task_type"] == "classification"
        assert out["num_predictions"] == 2


class TestPredict:
    async def test_by_reference_decodes_with_label_classes(self, monkeypatch):
        monkeypatch.setattr(neuralk_sdk, "build_client", lambda *a, **k: "CLIENT")
        monkeypatch.setattr(
            neuralk_sdk, "predict_by_reference",
            lambda client, key: {"predictions": [1, 0], "model": "seldon-small", "request_id": "r1"},
        )
        config = SeldonConfig(neuralk_api_key="nk_test", skip_api_key_validation=True)
        out = json.loads(await predict(_ctx(config), dataset_key="ds1", label_classes=["cat", "dog"]))
        assert out["predictions"] == ["dog", "cat"]
        assert out["dataset_id"] == "ds1"

    async def test_terms_not_accepted_surfaced(self, monkeypatch):
        from http import HTTPStatus

        from neuralk.exceptions import NeuralkTermsNotAcceptedError

        def raise_terms(client, key):
            raise NeuralkTermsNotAcceptedError(
                "terms", HTTPStatus.FORBIDDEN, "not accepted",
                terms_version="v1.0", terms_url="https://neuralk.ai/terms",
            )

        monkeypatch.setattr(neuralk_sdk, "build_client", lambda *a, **k: "CLIENT")
        monkeypatch.setattr(neuralk_sdk, "predict_by_reference", raise_terms)
        config = SeldonConfig(neuralk_api_key="nk_test", skip_api_key_validation=True)
        out = json.loads(await predict(_ctx(config), dataset_key="ds1"))
        assert "Terms of Service" in out["error"]
        assert out["terms_version"] == "v1.0"

    async def test_regression_path_floats_passthrough(self, monkeypatch):
        monkeypatch.setattr(neuralk_sdk, "build_client", lambda *a, **k: "CLIENT")
        monkeypatch.setattr(
            neuralk_sdk, "predict_inline",
            lambda client, **k: {"predictions": [2.5, 3.5], "probabilities": None, "model": "seldon-small"},
        )
        csv = "x,y\n1,10.5\n2,20.3\n3,30.7\n4,40.1\n5,50.9\n6,60.2"
        config = SeldonConfig(neuralk_api_key="nk_test", skip_api_key_validation=True)
        out = json.loads(await predict_from_data(
            _ctx(config), data=csv, target_column="y", task_type="regression",
        ))
        assert out["task_type"] == "regression"
        assert out["predictions"] == [2.5, 3.5]  # floats returned unchanged (no label decode)

    async def test_truncation_note_added(self, monkeypatch):
        monkeypatch.setattr(neuralk_sdk, "build_client", lambda *a, **k: "CLIENT")
        monkeypatch.setattr(
            neuralk_sdk, "predict_by_reference",
            lambda client, key: {"predictions": [1, 0, 1], "model": "m"},
        )
        config = SeldonConfig(neuralk_api_key="nk_test", skip_api_key_validation=True)
        out = json.loads(await predict(_ctx(config), dataset_key="ds1", max_predictions=1))
        assert out["num_predictions"] == 3
        assert len(out["predictions"]) == 1
        assert "note" in out

    async def test_sdk_exception_redacts_key(self, monkeypatch):
        monkeypatch.setattr(neuralk_sdk, "build_client", lambda *a, **k: "CLIENT")

        def raise_ex(client, key):
            raise NeuralkException("auth failed for nk_secret_key", 500, "detail")

        monkeypatch.setattr(neuralk_sdk, "predict_by_reference", raise_ex)
        config = SeldonConfig(neuralk_api_key="nk_secret_key", skip_api_key_validation=True)
        out = json.loads(await predict(_ctx(config), dataset_key="ds1"))
        assert "nk_secret_key" not in out["error"]
        assert "***" in out["error"]


# --- API-key validation gate (skip_api_key_validation=False) ---


class TestAuthValidationGate:
    async def test_rejected_key_surfaced_as_error(self, monkeypatch):
        async def raise_auth(*a, **k):
            raise APIKeyAuthError("Neuralk API key was rejected (401).")

        monkeypatch.setattr(server, "validate_api_key", raise_auth)
        config = SeldonConfig(neuralk_api_key="nk_test", skip_api_key_validation=False)
        out = json.loads(await predict(_ctx(config), dataset_key="ds1"))
        assert "rejected (401)" in out["error"]

    async def test_skip_validation_signal_proceeds(self, monkeypatch):
        async def raise_skip(*a, **k):
            raise _SkipValidation()

        monkeypatch.setattr(server, "validate_api_key", raise_skip)
        monkeypatch.setattr(neuralk_sdk, "build_client", lambda *a, **k: "CLIENT")
        monkeypatch.setattr(
            neuralk_sdk, "predict_by_reference", lambda client, key: {"predictions": [1], "model": "m"},
        )
        config = SeldonConfig(neuralk_api_key="nk_test", skip_api_key_validation=False)
        out = json.loads(await predict(_ctx(config), dataset_key="ds1"))
        assert out["predictions"] == [1]  # fail-open: proceeded to the (mocked) SDK call


# --- Presigned upload tools ---


class TestPresignedTools:
    async def test_create_upload_returns_recipe_and_parts(self, monkeypatch):
        async def fake_init(**k):
            return {"upload_id": "u1", "key": k["key"]}

        async def fake_sign(**k):
            return {"parts": [{"part_number": 1, "url": "https://s3.example/put?sig=x"}],
                    "expires_seconds": 3600}

        monkeypatch.setattr(server, "multipart_init", fake_init)
        monkeypatch.setattr(server, "multipart_sign", fake_sign)
        config = SeldonConfig(neuralk_api_key="nk_test", skip_api_key_validation=True)
        out = json.loads(await create_upload(_ctx(config)))
        assert out["upload_id"] == "u1"
        assert out["parts"][0]["url"].startswith("https://s3.example/put")
        assert "https://s3.example/put" in out["recipe"]  # presigned URL injected into recipe
        assert out["dataset_key"].endswith(".tar.zst")

    async def test_create_upload_403_gets_terms_hint(self, monkeypatch):
        from seldon_mcp.prediction_client import PredictionAPIError

        async def fake_init(**k):
            raise PredictionAPIError(403, "forbidden")

        monkeypatch.setattr(server, "multipart_init", fake_init)
        config = SeldonConfig(neuralk_api_key="nk_test", skip_api_key_validation=True)
        out = json.loads(await create_upload(_ctx(config)))
        assert "403" in out["error"]
        assert "Terms of Service" in out["error"]

    async def test_complete_upload(self, monkeypatch):
        async def fake_complete(**k):
            return {"key": k["key"], "location": "s3://bucket/x"}

        monkeypatch.setattr(server, "multipart_complete", fake_complete)
        config = SeldonConfig(neuralk_api_key="nk_test", skip_api_key_validation=True)
        out = json.loads(await complete_upload(
            _ctx(config), upload_id="u1", dataset_key="auto/x.tar.zst",
            parts=[{"part_number": 1, "etag": "e1"}],
        ))
        assert out["dataset_key"] == "auto/x.tar.zst"
        assert out["location"] == "s3://bucket/x"


# --- HTTP-only paths: download link + header key resolution ---


class TestAttachDownload:
    async def test_download_url_formed_over_http(self, monkeypatch, tmp_path):
        monkeypatch.setattr(neuralk_sdk, "build_client", lambda *a, **k: "CLIENT")
        monkeypatch.setattr(
            neuralk_sdk, "predict_by_reference", lambda client, key: {"predictions": [1, 0], "model": "m"},
        )
        monkeypatch.setattr(server, "_download_store", DownloadStore(tmp_path))
        config = SeldonConfig(neuralk_api_key="nk_test", skip_api_key_validation=True)
        ctx = _ctx(config, request=_http_request())
        out = json.loads(await predict(ctx, dataset_key="ds1"))
        assert "/downloads/" in out["download_url"]
        assert out["download_note"]


class TestKeyResolution:
    def test_header_key_beats_env(self):
        config = SeldonConfig(neuralk_api_key="env_key")
        ctx = _ctx(config, request=_http_request(headers={"x-neuralk-api-key": "hdr_key"}))
        assert _resolve_api_key(ctx, config) == "hdr_key"

    def test_env_fallback_when_no_header(self):
        config = SeldonConfig(neuralk_api_key="env_key")
        assert _resolve_api_key(_ctx(config), config) == "env_key"

    def test_no_key_raises(self):
        config = SeldonConfig(neuralk_api_key=None)
        try:
            _resolve_api_key(_ctx(config), config)
        except ValueError as e:
            assert "No Neuralk API key" in str(e)
        else:
            raise AssertionError("expected ValueError")


