from __future__ import annotations

from seldon_mcp.config import SeldonConfig


class TestSeldonConfig:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("NEURALK_API_KEY", raising=False)
        monkeypatch.delenv("NEURALK_HOST", raising=False)
        monkeypatch.delenv("SELDON_DEFAULT_MODEL", raising=False)
        monkeypatch.delenv("SELDON_DATA_DIR", raising=False)
        config = SeldonConfig(_env_file=None)

        assert config.neuralk_api_key is None
        assert config.neuralk_host is None
        assert config.seldon_default_model == "seldon-small"
        assert config.seldon_data_dir == "."

    def test_from_env_vars(self, monkeypatch):
        monkeypatch.setenv("NEURALK_API_KEY", "nk_test_123")
        monkeypatch.setenv("NEURALK_HOST", "http://localhost:9000")
        monkeypatch.setenv("SELDON_DEFAULT_MODEL", "seldon-large")
        monkeypatch.setenv("SELDON_DATA_DIR", "/data")
        config = SeldonConfig(_env_file=None)

        assert config.neuralk_api_key == "nk_test_123"
        assert config.neuralk_host == "http://localhost:9000"
        assert config.seldon_default_model == "seldon-large"
        assert config.seldon_data_dir == "/data"

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
