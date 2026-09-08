"""Async HTTP client for Neuralk's presigned multipart upload flow.

The inline-data and upload-by-reference prediction paths now go through the
official ``neuralk`` SDK (see ``neuralk_sdk.py``). What the SDK does NOT offer is
the presigned multipart upload: handing a client (e.g. a code-execution sandbox)
short-lived URLs so it PUTs the archive bytes directly to object storage, without
the Neuralk API key and without the data passing through this server. That flow
is implemented here with ``httpx`` against the prediction API:

  ``/uploads/multipart/init`` -> ``/sign`` -> ``/complete`` -> dataset_key

Only init/sign/complete (run server-side) use the key; the presigned PUT URLs do
not. The resulting key is passed to ``predict(dataset_key=...)``.

Authentication note: outbound requests use ``Authorization: Bearer <key>``. This
is distinct from the MCP server's own inbound per-request header
(``x-neuralk-api-key``); the resolved key value is the same secret, only the
header differs.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

ACCEPT_JSON = "application/json"

# Generous timeout: the API calls here are metadata-only, but keep headroom.
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=10.0)


class PredictionAPIError(Exception):
    """Raised when the prediction API returns a non-success response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{status_code}] {detail}")


def _auth_headers(api_key: str, *, content_type: str | None = None, accept: str = ACCEPT_JSON) -> dict[str, str]:
    """Build outbound headers for the prediction API."""
    headers = {"Authorization": f"Bearer {api_key}", "Accept": accept}
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _error_detail(response: httpx.Response) -> str:
    """Extract a human-readable error message from a failed response."""
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        return response.text or response.reason_phrase
    if isinstance(body, dict):
        for key in ("detail", "message", "error"):
            if key in body:
                return str(body[key])
    return json.dumps(body)


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    raise PredictionAPIError(response.status_code, _error_detail(response))


class _ClientContext:
    """Yield the provided client, or create (and close) a temporary one."""

    def __init__(self, client: httpx.AsyncClient | None, timeout: httpx.Timeout) -> None:
        self._provided = client
        self._timeout = timeout
        self._owned: httpx.AsyncClient | None = None

    async def __aenter__(self) -> httpx.AsyncClient:
        if self._provided is not None:
            return self._provided
        self._owned = httpx.AsyncClient(timeout=self._timeout)
        return self._owned

    async def __aexit__(self, *exc: object) -> None:
        if self._owned is not None:
            await self._owned.aclose()


# --- Presigned multipart upload ---
#
# Request/response shapes below were verified against the live Neuralk API. The
# presigned URLs let a client (e.g. a code-execution sandbox) upload bytes
# directly to object storage WITHOUT the Neuralk key — only init/sign/complete
# (run server-side) use the key.


async def multipart_init(
    *, base_url: str, api_key: str, key: str, client: httpx.AsyncClient | None = None
) -> dict[str, Any]:
    """Start a multipart upload for object ``key``. Returns ``{upload_id, key}``."""
    url = f"{base_url.rstrip('/')}/api/v1/uploads/multipart/init"
    headers = _auth_headers(api_key, content_type=ACCEPT_JSON)
    async with _ClientContext(client, DEFAULT_TIMEOUT) as c:
        response = await c.post(url, json={"key": key}, headers=headers)
    _raise_for_status(response)
    return response.json()


async def multipart_sign(
    *,
    base_url: str,
    api_key: str,
    upload_id: str,
    key: str,
    part_count: int,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Get presigned PUT URLs for ``part_count`` parts.

    Returns ``{upload_id, key, parts: [{part_number, url}], expires_seconds}``.
    """
    url = f"{base_url.rstrip('/')}/api/v1/uploads/multipart/sign"
    headers = _auth_headers(api_key, content_type=ACCEPT_JSON)
    payload = {"upload_id": upload_id, "key": key, "part_count": part_count}
    async with _ClientContext(client, DEFAULT_TIMEOUT) as c:
        response = await c.post(url, json=payload, headers=headers)
    _raise_for_status(response)
    return response.json()


async def multipart_complete(
    *,
    base_url: str,
    api_key: str,
    upload_id: str,
    key: str,
    parts: list[dict[str, Any]],
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Finalize the multipart upload.

    ``parts`` is an ordered list of ``{"part_number": int, "etag": str}``.
    Returns ``{key, etag, location}``; ``key`` is the dataset_key for inference.
    """
    url = f"{base_url.rstrip('/')}/api/v1/uploads/multipart/complete"
    headers = _auth_headers(api_key, content_type=ACCEPT_JSON)
    payload = {"upload_id": upload_id, "key": key, "parts": parts}
    async with _ClientContext(client, DEFAULT_TIMEOUT) as c:
        response = await c.post(url, json=payload, headers=headers)
    _raise_for_status(response)
    return response.json()
