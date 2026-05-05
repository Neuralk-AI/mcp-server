"""Tests for seldon_mcp.auth.validate_api_key."""

from __future__ import annotations

import httpx
import pytest

from seldon_mcp import auth as auth_mod
from seldon_mcp.auth import APIKeyAuthError, _SkipValidation, validate_api_key


@pytest.fixture(autouse=True)
def _reset_cache():
    auth_mod.reset_cache()
    yield
    auth_mod.reset_cache()


def _mock_transport(handler):
    return httpx.MockTransport(handler)


@pytest.fixture
def patch_client(monkeypatch):
    """Patch httpx.AsyncClient with a MockTransport-backed client."""

    def _patch(handler):
        original = httpx.AsyncClient

        def factory(*args, **kwargs):
            kwargs["transport"] = _mock_transport(handler)
            return original(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", factory)

    return _patch


@pytest.mark.asyncio
async def test_validate_returns_whoami_payload(patch_client):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/auth/whoami"
        assert request.headers["authorization"] == "Bearer nk_live_good"
        return httpx.Response(
            200,
            json={
                "organization_id": "11111111-1111-1111-1111-111111111111",
                "key_id": "22222222-2222-2222-2222-222222222222",
                "key_name": "default",
                "key_type": "live",
                "scopes": ["read"],
                "is_active": True,
            },
        )

    patch_client(handler)

    result = await validate_api_key("nk_live_good", "https://api.example.com")
    assert result.organization_id == "11111111-1111-1111-1111-111111111111"
    assert result.scopes == ("read",)
    assert result.is_active is True


@pytest.mark.asyncio
async def test_validate_caches_success(patch_client):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            200,
            json={
                "organization_id": "11111111-1111-1111-1111-111111111111",
                "key_id": "22222222-2222-2222-2222-222222222222",
                "key_name": "k",
                "key_type": "live",
                "scopes": ["read"],
                "is_active": True,
            },
        )

    patch_client(handler)

    await validate_api_key("nk_live_good", "https://api.example.com")
    await validate_api_key("nk_live_good", "https://api.example.com")
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_validate_raises_on_401(patch_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": {"message": "Invalid API key"}})

    patch_client(handler)

    with pytest.raises(APIKeyAuthError, match="rejected"):
        await validate_api_key("nk_live_bad", "https://api.example.com")


@pytest.mark.asyncio
async def test_validate_skips_on_network_error(patch_client):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    patch_client(handler)

    with pytest.raises(_SkipValidation):
        await validate_api_key("nk_live_good", "https://api.example.com")


@pytest.mark.asyncio
async def test_validate_skips_on_5xx(patch_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream down")

    patch_client(handler)

    with pytest.raises(_SkipValidation):
        await validate_api_key("nk_live_good", "https://api.example.com")
