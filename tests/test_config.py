from __future__ import annotations

from seldon_mcp.config import SeldonConfig


class TestSeldonConfig:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("NEURALK_API_KEY", raising=False)
        monkeypatch.delenv("NEURALK_PREDICTION_URL", raising=False)
        monkeypatch.delenv("SELDON_DEFAULT_MODEL", raising=False)
        config = SeldonConfig(_env_file=None)

        assert config.neuralk_api_key is None
        assert config.neuralk_prediction_url == "https://api.prediction.neuralk-ai.com"
        assert config.seldon_default_model == "seldon-small"

    def test_from_env_vars(self, monkeypatch):
        monkeypatch.setenv("NEURALK_API_KEY", "nk_test_123")
        monkeypatch.setenv("NEURALK_PREDICTION_URL", "https://example.test")
        monkeypatch.setenv("SELDON_DEFAULT_MODEL", "seldon-large")
        config = SeldonConfig(_env_file=None)

        assert config.neuralk_api_key == "nk_test_123"
        assert config.neuralk_prediction_url == "https://example.test"
        assert config.seldon_default_model == "seldon-large"

    def test_api_key_optional(self, monkeypatch):
        monkeypatch.delenv("NEURALK_API_KEY", raising=False)
        config = SeldonConfig(_env_file=None)
        assert config.neuralk_api_key is None

    def test_env_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NEURALK_API_KEY", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("NEURALK_API_KEY=nk_from_file\n")
        config = SeldonConfig(_env_file=str(env_file))
        assert config.neuralk_api_key == "nk_from_file"
