"""API key validation against the neuralk-saas auth API.

Calls ``GET {base_url}/api/v1/auth/whoami`` to confirm a key is active before it
is handed off to the Seldon SDK. Results are cached in-process with a short TTL
so repeated tool calls don't hammer the auth endpoint.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger("seldon_mcp.auth")


class APIKeyAuthError(Exception):
    """Raised when an API key is rejected by the neuralk-saas auth API."""


@dataclass(frozen=True)
class WhoAmIResult:
    organization_id: str
    key_id: str
    key_name: str
    key_type: str
    scopes: tuple[str, ...]
    is_active: bool


@dataclass
class _CacheEntry:
    result: WhoAmIResult
    expires_at: float


_cache: dict[str, _CacheEntry] = {}


def _cache_key(api_key: str, base_url: str) -> str:
    digest = hashlib.sha256(f"{base_url}|{api_key}".encode()).hexdigest()
    return digest


def _whoami_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/api/v1/auth/whoami"


async def validate_api_key(
    api_key: str,
    base_url: str,
    *,
    ttl_s: int = 300,
    timeout_s: float = 5.0,
) -> WhoAmIResult:
    """Validate an API key against neuralk-saas.

    On success, returns the whoami payload and caches it for ``ttl_s`` seconds.
    On 401 the cache is bypassed and ``APIKeyAuthError`` is raised. On network
    or 5xx errors we fail-open: the validation is skipped and the SDK call will
    surface the real error itself.
    """
    cache_key = _cache_key(api_key, base_url)
    now = time.monotonic()
    entry = _cache.get(cache_key)
    if entry is not None and entry.expires_at > now:
        return entry.result

    url = _whoami_url(base_url)
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
    except httpx.HTTPError as exc:
        logger.warning("whoami unreachable at %s (%s); skipping pre-validation", url, exc)
        raise _SkipValidation() from exc

    if response.status_code == 401:
        _cache.pop(cache_key, None)
        raise APIKeyAuthError(
            "Neuralk API key was rejected (401). The key is missing, invalid, "
            "or has been revoked. Generate a new key at "
            "https://prediction.neuralk-ai.com/dashboard/api-keys."
        )
    if response.status_code == 403:
        _cache.pop(cache_key, None)
        try:
            detail = response.json()
        except ValueError:
            detail = {}
        message = (
            detail.get("detail", {}).get("message")
            if isinstance(detail.get("detail"), dict)
            else None
        ) or (
            "Neuralk API key was rejected (403). The organization may be expired "
            "or the terms of service have not been accepted."
        )
        raise APIKeyAuthError(message)
    if response.status_code >= 500:
        logger.warning(
            "whoami returned %s at %s; skipping pre-validation", response.status_code, url
        )
        raise _SkipValidation()
    if response.status_code != 200:
        raise APIKeyAuthError(
            f"Unexpected response from {url}: HTTP {response.status_code}"
        )

    data = response.json()
    result = WhoAmIResult(
        organization_id=str(data["organization_id"]),
        key_id=str(data["key_id"]),
        key_name=str(data.get("key_name", "")),
        key_type=str(data.get("key_type", "")),
        scopes=tuple(data.get("scopes") or ()),
        is_active=bool(data.get("is_active", True)),
    )
    if not result.is_active:
        raise APIKeyAuthError("API key is no longer active.")

    _cache[cache_key] = _CacheEntry(result=result, expires_at=now + ttl_s)
    return result


class _SkipValidation(Exception):
    """Internal signal that pre-validation is unavailable and should be bypassed."""


def reset_cache() -> None:
    """Clear the validation cache. Intended for tests."""
    _cache.clear()
