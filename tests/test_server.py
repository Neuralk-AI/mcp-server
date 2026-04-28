from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from seldon_mcp.server import (
    SELDON_MODELS,
    VALID_MODEL_NAMES,
    VALID_TASK_TYPES,
    _get_task_type,
    _sanitize_error,
    _validate_params,
)

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


# --- _get_task_type ---


class TestGetTaskType:
    def test_from_task_type_attr(self):
        model = MagicMock()
        model.task_type_ = "classification"
        assert _get_task_type(model) == "classification"

    def test_from_classes_attr(self):
        model = MagicMock(spec=["classes_"])
        del model.task_type_
        model.classes_ = ["a", "b"]
        assert _get_task_type(model) == "classification"

    def test_fallback_regression(self):
        model = MagicMock(spec=[])
        assert _get_task_type(model) == "regression"


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
        assert tool_names == {"describe_data", "predict", "evaluate", "list_models"}

    def test_resources_registered(self):
        from seldon_mcp.server import mcp

        resource_uris = {str(r.uri) for r in mcp._resource_manager.list_resources()}
        assert "seldon://models" in resource_uris
        assert "seldon://config" in resource_uris

    def test_prompts_registered(self):
        from seldon_mcp.server import mcp

        prompt_names = {p.name for p in mcp._prompt_manager.list_prompts()}
        assert prompt_names == {"classify", "regress", "compare_models"}

    def test_server_has_instructions(self):
        from seldon_mcp.server import mcp

        assert mcp.instructions is not None
        assert "in-context learning" in mcp.instructions
        assert "context selection" in mcp.instructions.upper() or "context selection" in mcp.instructions
