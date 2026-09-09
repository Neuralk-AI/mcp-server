"""The hosted (Streamable HTTP) surface: key headers, the 401 gate, probes,
public URL of download links, and the process-wide state that must survive a
session ending."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from seldon_mcp import neuralk_sdk, server
from seldon_mcp.config import SeldonConfig
from seldon_mcp.server import (
    RequireClientKeyMiddleware,
    _api_key_from_headers,
    _configure_binding,
    _init_state,
    _resolve_api_key,
    app_lifespan,
    build_http_app,
    mcp,
    predict,
)


def _ctx(config: SeldonConfig, request=None):
    rc = SimpleNamespace(lifespan_context={"config": config}, request=request)
    return SimpleNamespace(request_context=rc)


def _http_request(headers=None, base_url="http://pod:8000/"):
    return SimpleNamespace(headers=headers or {}, base_url=base_url)


@pytest.fixture
def fresh_state(monkeypatch, tmp_path):
    """Module-level state as at process start, with downloads in tmp_path."""
    monkeypatch.setattr(server, "_lifespan_config", None)
    monkeypatch.setattr(server, "_download_store", None)
    monkeypatch.setattr(server, "_server_key_checked", False)
    return tmp_path


# --- key headers ---


class TestApiKeyFromHeaders:
    def test_x_neuralk_api_key_header(self):
        assert _api_key_from_headers({"x-neuralk-api-key": " nk_a "}) == "nk_a"

    def test_generic_x_api_key_header(self):
        # The header MCP hosting platforms (Alpic) forward for API-key auth.
        assert _api_key_from_headers({"x-api-key": " nk_c "}) == "nk_c"

    def test_neuralk_header_wins_over_generic(self):
        assert _api_key_from_headers({"x-neuralk-api-key": "nk_a", "x-api-key": "nk_c"}) == "nk_a"

    def test_authorization_bearer(self):
        assert _api_key_from_headers({"authorization": "Bearer nk_b"}) == "nk_b"

    def test_bearer_scheme_is_case_insensitive(self):
        assert _api_key_from_headers({"authorization": "bearer nk_b"}) == "nk_b"

    def test_x_header_beats_bearer(self):
        headers = {"x-neuralk-api-key": "nk_a", "authorization": "Bearer nk_b"}
        assert _api_key_from_headers(headers) == "nk_a"

    def test_other_scheme_ignored(self):
        assert _api_key_from_headers({"authorization": "Basic abc"}) is None

    def test_blank_values_are_no_key(self):
        assert _api_key_from_headers({"x-neuralk-api-key": "  ", "authorization": "Bearer  "}) is None

    def test_no_headers(self):
        assert _api_key_from_headers({}) is None


# --- hosted mode: the server's key is never used on a client's behalf ---


class TestRequireClientApiKey:
    def test_header_key_used(self):
        config = SeldonConfig(neuralk_api_key="server_key", require_client_api_key=True)
        ctx = _ctx(config, request=_http_request(headers={"authorization": "Bearer nk_client"}))
        assert _resolve_api_key(ctx, config) == "nk_client"

    def test_no_header_refused_even_with_server_key(self):
        config = SeldonConfig(neuralk_api_key="server_key", require_client_api_key=True)
        with pytest.raises(ValueError, match="your own Neuralk API key"):
            _resolve_api_key(_ctx(config, request=_http_request()), config)

    def test_stdio_without_header_refused(self):
        config = SeldonConfig(neuralk_api_key="server_key", require_client_api_key=True)
        with pytest.raises(ValueError, match="x-neuralk-api-key"):
            _resolve_api_key(_ctx(config), config)

    async def test_tool_reports_missing_key_as_error(self):
        config = SeldonConfig(neuralk_api_key="server_key", require_client_api_key=True, skip_api_key_validation=True)
        out = json.loads(await predict(_ctx(config, request=_http_request()), dataset_key="ds1"))
        assert "your own Neuralk API key" in out["error"]
        assert "server_key" not in out["error"]


# --- the 401 gate ---


def _gated_app() -> Starlette:
    async def ok(request):
        return PlainTextResponse("served")

    app = Starlette(
        routes=[
            Route("/mcp", ok, methods=["GET", "POST"]),
            Route("/", ok, methods=["GET", "POST"]),
            Route("/healthz", ok, methods=["GET"]),
            Route("/downloads/{token}", ok, methods=["GET"]),
        ]
    )
    app.add_middleware(RequireClientKeyMiddleware, mcp_paths={"/mcp", "/"})
    return app


CALL = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "list_models", "arguments": {}}}
INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}}
LIST = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}


class TestRequireClientKeyMiddleware:
    def test_tool_call_without_key_is_401_with_challenge(self):
        response = TestClient(_gated_app()).post("/mcp", json=CALL)
        assert response.status_code == 401
        assert response.headers["www-authenticate"].startswith("Bearer")
        assert "x-neuralk-api-key" in response.json()["error"]

    def test_discovery_without_key_passes(self):
        client = TestClient(_gated_app())
        for body in (INIT, LIST, {"jsonrpc": "2.0", "method": "notifications/initialized"}):
            response = client.post("/mcp", json=body)
            assert response.status_code == 200, body
            assert response.text == "served"

    def test_batch_with_a_tool_call_is_gated(self):
        assert TestClient(_gated_app()).post("/mcp", json=[LIST, CALL]).status_code == 401

    def test_unparseable_body_is_left_to_the_transport(self):
        assert TestClient(_gated_app()).post("/mcp", content=b"not json").status_code == 200

    def test_root_without_key_is_gated(self):
        assert TestClient(_gated_app()).post("/", json=CALL).status_code == 401

    def test_trailing_slash_is_gated_too(self):
        assert TestClient(_gated_app()).post("/mcp/", json=CALL).status_code == 401

    def test_tool_call_with_x_header_passes(self):
        response = TestClient(_gated_app()).post("/mcp", json=CALL, headers={"x-neuralk-api-key": "nk_a"})
        assert response.status_code == 200
        assert response.text == "served"

    def test_tool_call_with_bearer_passes(self):
        response = TestClient(_gated_app()).post("/mcp", json=CALL, headers={"Authorization": "Bearer nk_a"})
        assert response.status_code == 200

# --- the built app ---


class TestBuildHttpApp:
    def test_serves_probe_and_gates_mcp(self, fresh_state):
        _init_state(SeldonConfig(require_client_api_key=True, seldon_download_dir=str(fresh_state)))
        app = build_http_app("0.0.0.0", 8000)
        client = TestClient(app)  # no lifespan: the routes alone are under test
        assert client.get("/healthz").text == "ok"
        assert client.post("/mcp", json=CALL).status_code == 401
        assert client.post("/", json=CALL).status_code == 401
        assert client.get("/downloads/nope").status_code == 404

    def test_non_loopback_bind_drops_dns_rebinding_protection(self):
        _configure_binding("0.0.0.0", 8000)
        assert mcp.settings.transport_security is not None
        assert mcp.settings.transport_security.enable_dns_rebinding_protection is False

    def test_stateless_and_json_settings_applied(self, fresh_state):
        _init_state(SeldonConfig(seldon_download_dir=str(fresh_state)))
        build_http_app("0.0.0.0", 8000, stateless=True, json_response=True)
        assert mcp.settings.stateless_http is True
        assert mcp.settings.json_response is True


# --- process-wide state ---


class TestProcessState:
    def test_init_state_is_idempotent(self, fresh_state):
        first = _init_state(SeldonConfig(seldon_download_dir=str(fresh_state)))
        store = server._download_store
        second = _init_state(SeldonConfig(seldon_download_dir="/elsewhere"))
        assert second is first
        assert server._download_store is store

    async def test_session_end_keeps_the_store_for_other_sessions(self, fresh_state):
        _init_state(SeldonConfig(seldon_download_dir=str(fresh_state)))
        async with app_lifespan(mcp) as first:
            store = server._download_store
            assert store is not None
        # A second session (or the next stateless request) finds the same store.
        async with app_lifespan(mcp) as second:
            assert server._download_store is store
            assert second["config"] is first["config"]
        assert server._download_store is store

    async def test_server_key_checked_once_per_process(self, fresh_state, monkeypatch):
        calls = []

        async def fake_validate(*a, **k):
            calls.append(1)
            raise server._SkipValidation()

        monkeypatch.setattr(server, "validate_api_key", fake_validate)
        _init_state(SeldonConfig(neuralk_api_key="nk_server", seldon_download_dir=str(fresh_state)))
        async with app_lifespan(mcp):
            pass
        async with app_lifespan(mcp):
            pass
        assert len(calls) == 1


# --- download links behind an ingress ---


class TestPublicUrl:
    async def test_download_link_uses_public_url(self, fresh_state, monkeypatch):
        monkeypatch.setattr(neuralk_sdk, "build_client", lambda *a, **k: "CLIENT")
        monkeypatch.setattr(
            neuralk_sdk,
            "predict_by_reference",
            lambda client, key: {"predictions": [1, 0], "model": "m"},
        )
        config = SeldonConfig(
            neuralk_api_key="nk_test",
            skip_api_key_validation=True,
            seldon_public_url="https://mcp.example/",
            seldon_download_dir=str(fresh_state),
        )
        _init_state(config)
        out = json.loads(await predict(_ctx(config, request=_http_request()), dataset_key="ds1"))
        assert out["download_url"].startswith("https://mcp.example/downloads/")

    async def test_download_link_falls_back_to_request_base_url(self, fresh_state, monkeypatch):
        monkeypatch.setattr(neuralk_sdk, "build_client", lambda *a, **k: "CLIENT")
        monkeypatch.setattr(
            neuralk_sdk,
            "predict_by_reference",
            lambda client, key: {"predictions": [1, 0], "model": "m"},
        )
        config = SeldonConfig(
            neuralk_api_key="nk_test",
            skip_api_key_validation=True,
            seldon_download_dir=str(fresh_state),
        )
        _init_state(config)
        request = _http_request(base_url="http://pod:8000/")
        out = json.loads(await predict(_ctx(config, request=request), dataset_key="ds1"))
        assert out["download_url"].startswith("http://pod:8000/downloads/")
