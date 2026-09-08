from __future__ import annotations

import json

from seldon_mcp.prediction_client import (
    ACCEPT_JSON,
    _auth_headers,
    multipart_complete,
    multipart_init,
    multipart_sign,
)

BASE = "https://api.example.com"
KEY = "nk_live_secret"


# --- _auth_headers ---


class TestAuthHeaders:
    def test_bearer_and_default_accept(self):
        h = _auth_headers(KEY)
        assert h["Authorization"] == f"Bearer {KEY}"
        assert h["Accept"] == ACCEPT_JSON
        assert "Content-Type" not in h

    def test_content_type(self):
        h = _auth_headers(KEY, content_type=ACCEPT_JSON)
        assert h["Content-Type"] == ACCEPT_JSON


# --- presigned multipart (verified shapes) ---


class TestMultipart:
    async def test_init(self, httpx_mock):
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/uploads/multipart/init", method="POST",
            status_code=201, json={"upload_id": "u1", "key": "auto/x.tar.zst"},
        )
        out = await multipart_init(base_url=BASE, api_key=KEY, key="auto/x.tar.zst")
        assert out == {"upload_id": "u1", "key": "auto/x.tar.zst"}
        req = httpx_mock.get_request()
        assert json.loads(req.content) == {"key": "auto/x.tar.zst"}
        assert req.headers["Authorization"] == f"Bearer {KEY}"

    async def test_sign_returns_presigned_parts(self, httpx_mock):
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/uploads/multipart/sign", method="POST",
            json={"upload_id": "u1", "key": "auto/x.tar.zst",
                  "parts": [{"part_number": 1, "url": "https://s3.example/put?sig=abc"}],
                  "expires_seconds": 21600},
        )
        out = await multipart_sign(base_url=BASE, api_key=KEY, upload_id="u1", key="auto/x.tar.zst", part_count=1)
        assert out["parts"][0]["url"].startswith("https://s3.example/put")
        assert json.loads(httpx_mock.get_request().content) == {
            "upload_id": "u1", "key": "auto/x.tar.zst", "part_count": 1,
        }

    async def test_complete(self, httpx_mock):
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/uploads/multipart/complete", method="POST",
            json={"key": "auto/x.tar.zst", "etag": "abc-1", "location": "https://s3.example/x"},
        )
        out = await multipart_complete(
            base_url=BASE, api_key=KEY, upload_id="u1", key="auto/x.tar.zst",
            parts=[{"part_number": 1, "etag": '"etag1"'}],
        )
        assert out["key"] == "auto/x.tar.zst"
        body = json.loads(httpx_mock.get_request().content)
        assert body["parts"] == [{"part_number": 1, "etag": '"etag1"'}]
