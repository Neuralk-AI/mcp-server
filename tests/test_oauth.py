"""OAuth sign-in: the flow an MCP client (Claude) runs against the hosted server,
end to end, with the Neuralk APIs (whoami, Keycloak, API keys) mocked."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from datetime import datetime
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from seldon_mcp import auth as auth_mod
from seldon_mcp import oauth as oauth_mod
from seldon_mcp import server
from seldon_mcp.config import SeldonConfig
from seldon_mcp.oauth import OAuthServer, Sealer, redirect_uri_matches
from seldon_mcp.server import RequireClientKeyMiddleware, _api_key_from_headers, _init_state, build_http_app

SECRET = "x" * 48
PUBLIC = "https://mcp.example"
API = "https://api.test"
KC = "https://auth.test/realms/Neuralk"
REDIRECT = "https://claude.ai/api/mcp/auth_callback"
VERIFIER = "v" * 64
CHALLENGE = base64.urlsafe_b64encode(hashlib.sha256(VERIFIER.encode()).digest()).decode().rstrip("=")
GOOD_KEY = "nk_live_good"
INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}}


def _config(**overrides) -> SeldonConfig:
    values = {
        "seldon_oauth_secret": SECRET,
        "seldon_public_url": PUBLIC,
        "require_client_api_key": True,
        "neuralk_prediction_url": API,
        "neuralk_oidc_issuer": KC,
    }
    values.update(overrides)
    return SeldonConfig(**values)


class FakeNeuralk:
    """whoami, the Keycloak realm and the API-key endpoint, behind one MockTransport."""

    def __init__(self) -> None:
        self.valid_keys = {GOOD_KEY}
        self.create_status = 201
        self.create_body: dict = {"api_key": "nk_live_created", "key": {}}
        self.created: list[dict] = []
        self.kc_token_forms: list[dict] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == f"{API}/api/v1/auth/whoami":
            key = request.headers["authorization"].removeprefix("Bearer ")
            if key not in self.valid_keys:
                return httpx.Response(401, json={"detail": "Invalid API key"})
            return httpx.Response(
                200, json={"organization_id": "org", "key_id": "k", "key_name": "n", "scopes": ["read", "write"]}
            )
        if url == f"{KC}/.well-known/openid-configuration":
            return httpx.Response(
                200,
                json={
                    "authorization_endpoint": f"{KC}/protocol/openid-connect/auth",
                    "token_endpoint": f"{KC}/protocol/openid-connect/token",
                },
            )
        if url == f"{KC}/protocol/openid-connect/token":
            form = dict(parse_qs(request.content.decode()))
            self.kc_token_forms.append({k: v[0] for k, v in form.items()})
            return httpx.Response(200, json={"access_token": "kc-user-token", "token_type": "Bearer"})
        if url == f"{API}/api/v1/api-keys" and request.method == "POST":
            assert request.headers["authorization"] == "Bearer kc-user-token"
            self.created.append(json.loads(request.content))
            if self.create_status == 201:
                self.valid_keys.add(self.create_body["api_key"])
            return httpx.Response(self.create_status, json=self.create_body)
        return httpx.Response(404)


@pytest.fixture
def neuralk(monkeypatch):
    fake = FakeNeuralk()
    original = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(fake.handle)
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    auth_mod.reset_cache()
    yield fake
    auth_mod.reset_cache()


def _make_app(config: SeldonConfig, monkeypatch) -> tuple[TestClient, OAuthServer]:
    """The OAuth routes, the gate, and an /mcp endpoint that echoes the key the tools would get."""
    oauth = OAuthServer(config)
    monkeypatch.setattr(server, "_oauth", oauth)

    async def mcp_endpoint(request):
        return JSONResponse({"key": _api_key_from_headers(request.headers)})

    app = Starlette(
        routes=[
            Route("/mcp", mcp_endpoint, methods=["GET", "POST"]),
            Route("/", mcp_endpoint, methods=["GET", "POST"]),
            *oauth.routes(),
        ]
    )
    app.add_middleware(RequireClientKeyMiddleware, mcp_paths={"/mcp", "/"}, oauth=oauth)
    return TestClient(app, base_url=PUBLIC), oauth


@pytest.fixture
def app(monkeypatch, neuralk):
    client, _ = _make_app(_config(), monkeypatch)
    return client


@pytest.fixture
def app_with_neuralk_sign_in(monkeypatch, neuralk):
    client, _ = _make_app(
        _config(neuralk_oidc_client_id="seldon-mcp", neuralk_oidc_client_secret="kc-secret"), monkeypatch
    )
    return client


# --- the client's side of the flow, step by step ---


def _register(client: TestClient, **metadata) -> dict:
    body = {"redirect_uris": [REDIRECT], "token_endpoint_auth_method": "none", "client_name": "Claude"}
    body.update(metadata)
    response = client.post("/register", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def _authorize(client: TestClient, client_id: str, redirect_uri: str = REDIRECT, state: str = "st-1") -> str:
    response = client.get(
        "/authorize",
        params={
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "code_challenge": CHALLENGE,
            "code_challenge_method": "S256",
            "state": state,
            "scope": "seldon offline_access",
            "resource": f"{PUBLIC}/mcp",
        },
        follow_redirects=False,
    )
    assert response.status_code == 302, response.text
    location = response.headers["location"]
    assert location.startswith(f"{PUBLIC}/oauth/consent?request=")
    return parse_qs(urlsplit(location).query)["request"][0]


def _open_page(client: TestClient, pending: str) -> str:
    """Show the sign-in page, as the browser does (it sets the sign-in cookie); return its csrf field."""
    page = client.get("/oauth/consent", params={"request": pending})
    assert page.status_code == 200, page.text
    return re.search(r'name="csrf" value="([^"]+)"', page.text).group(1)


def _submit_key(client: TestClient, pending: str, key: str):
    csrf = _open_page(client, pending)
    return client.post(
        "/oauth/consent",
        data={"request": pending, "api_key": key, "action": "key", "csrf": csrf},
        follow_redirects=False,
    )


def _query(location: str) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(urlsplit(location).query).items()}


def _exchange(client: TestClient, client_id: str, code: str, **overrides):
    form = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT,
        "client_id": client_id,
        "code_verifier": VERIFIER,
    }
    form.update(overrides)
    return client.post("/token", data=form)


def _sign_in_with_key(client: TestClient, key: str = GOOD_KEY) -> tuple[dict, dict]:
    registered = _register(client)
    pending = _authorize(client, registered["client_id"])
    redirect = _submit_key(client, pending, key)
    assert redirect.status_code == 302, redirect.text
    code = _query(redirect.headers["location"])["code"]
    tokens = _exchange(client, registered["client_id"], code)
    assert tokens.status_code == 200, tokens.text
    return registered, tokens.json()


# --- discovery ---


class TestDiscovery:
    def test_request_without_credentials_gets_the_challenge(self, app):
        response = app.post("/mcp", json=INIT)
        assert response.status_code == 401
        challenge = response.headers["www-authenticate"]
        assert challenge.startswith("Bearer ")
        assert f'resource_metadata="{PUBLIC}/.well-known/oauth-protected-resource/mcp"' in challenge
        assert 'scope="seldon"' in challenge
        assert "error=" not in challenge  # no credentials is not an invalid token (RFC 6750 section 3.1)

    def test_get_and_root_are_gated_too(self, app):
        assert app.get("/mcp").status_code == 401
        root = app.post("/", json=INIT)
        assert root.status_code == 401
        assert f'resource_metadata="{PUBLIC}/.well-known/oauth-protected-resource"' in root.headers["www-authenticate"]

    def test_cors_preflight_passes(self, app):
        assert app.options("/mcp").status_code != 401

    def test_protected_resource_metadata(self, app):
        doc = app.get("/.well-known/oauth-protected-resource/mcp").json()
        assert doc["resource"] == f"{PUBLIC}/mcp"
        assert doc["authorization_servers"] == [PUBLIC]
        assert doc["scopes_supported"] == ["seldon"]
        assert app.get("/.well-known/oauth-protected-resource").json()["resource"] == PUBLIC

    def test_authorization_server_metadata(self, app):
        doc = app.get("/.well-known/oauth-authorization-server").json()
        assert doc["issuer"] == PUBLIC
        assert doc["authorization_endpoint"] == f"{PUBLIC}/authorize"
        assert doc["token_endpoint"] == f"{PUBLIC}/token"
        assert doc["registration_endpoint"] == f"{PUBLIC}/register"
        assert doc["code_challenge_methods_supported"] == ["S256"]
        assert "none" in doc["token_endpoint_auth_methods_supported"]

    def test_raw_api_key_headers_still_work(self, app):
        for headers in ({"x-neuralk-api-key": "nk_a"}, {"x-api-key": "nk_a"}, {"Authorization": "Bearer nk_a"}):
            response = app.post("/mcp", json=INIT, headers=headers)
            assert response.status_code == 200, headers
            assert response.json() == {"key": "nk_a"}


# --- registration ---


class TestRegistration:
    def test_public_client(self, app):
        registered = _register(app)
        assert registered["client_id"].startswith("seldon_client_")
        assert registered["token_endpoint_auth_method"] == "none"
        assert "client_secret" not in registered
        assert registered["redirect_uris"] == [REDIRECT]

    def test_confidential_client_gets_a_derived_secret(self, app):
        registered = _register(app, token_endpoint_auth_method="client_secret_basic")
        assert registered["token_endpoint_auth_method"] == "client_secret_post"
        assert registered["client_secret"]
        assert _register(app, token_endpoint_auth_method=None)["token_endpoint_auth_method"] == "client_secret_post"

    @pytest.mark.parametrize(
        "uri",
        [
            "http://evil.example/cb",
            "javascript://x/alert(1)",
            "https://claude.ai/cb#frag",
            # Schemes that open a web page: the code would land on a site.
            "x-safari-https://evil.example/cb",
            "googlechromes://evil.example/cb",
            "microsoft-edge:https://evil.example/cb",
            "com.evil.browser:https://evil.example/cb",
            "ftp://evil.example/cb",
            "wss://evil.example/cb",
        ],
    )
    def test_unsafe_redirect_uris_refused(self, app, uri):
        response = app.post("/register", json={"redirect_uris": [uri]})
        assert response.status_code == 400
        assert response.json()["error"] == "invalid_redirect_uri"

    @pytest.mark.parametrize(
        "uri",
        [
            "http://localhost:3118/callback",
            "http://127.0.0.1/callback",
            "cursor://anysphere.cursor-retrieval/oauth/callback",
            "vscode://vscode.github-authentication/did-authenticate",
            "com.example.app:/oauth2redirect",
        ],
    )
    def test_native_redirect_uris_accepted(self, app, uri):
        assert app.post("/register", json={"redirect_uris": [uri]}).status_code == 201

    def test_malformed_body(self, app):
        assert app.post("/register", content=b"nope").json()["error"] == "invalid_client_metadata"
        assert app.post("/register", json={"redirect_uris": []}).status_code == 400
        assert app.post("/register", json={"redirect_uris": None}).status_code == 400

    @pytest.mark.parametrize(
        "params",
        [
            {"response_type": "token", "code_challenge": CHALLENGE},
            {"response_type": "code"},
            {"response_type": "code", "code_challenge": CHALLENGE, "code_challenge_method": "plain"},
        ],
    )
    def test_bad_authorize_requests_are_not_redirected(self, app, params):
        # The SDK would send these errors to the registered redirect URI: an
        # open redirector, registration being open to anyone.
        registered = _register(app, redirect_uris=["https://evil.example/phish"])
        query = {"client_id": registered["client_id"], "redirect_uri": "https://evil.example/phish", **params}
        response = app.get("/authorize", params=query, follow_redirects=False)
        assert response.status_code == 400
        assert "location" not in response.headers
        assert "not valid" in response.text

    def test_unknown_client_cannot_authorize(self, app):
        response = app.get(
            "/authorize",
            params={"client_id": "someone", "response_type": "code", "code_challenge": CHALLENGE},
            follow_redirects=False,
        )
        assert response.status_code == 400

    def test_unregistered_redirect_refused(self, app):
        registered = _register(app)
        response = app.get(
            "/authorize",
            params={
                "client_id": registered["client_id"],
                "response_type": "code",
                "redirect_uri": "https://evil.example/cb",
                "code_challenge": CHALLENGE,
            },
            follow_redirects=False,
        )
        assert response.status_code == 400


# --- signing in with an API key ---


class TestKeySignIn:
    def test_full_flow_gives_tokens_that_carry_the_key(self, app):
        _, tokens = _sign_in_with_key(app)
        assert tokens["token_type"] == "Bearer"
        assert tokens["access_token"].startswith("seldon_at_")
        assert tokens["refresh_token"].startswith("seldon_rt_")
        assert tokens["expires_in"] == oauth_mod.ACCESS_TOKEN_TTL_S
        assert tokens["scope"] == "seldon"
        response = app.post("/mcp", json=INIT, headers={"Authorization": f"Bearer {tokens['access_token']}"})
        assert response.status_code == 200
        assert response.json() == {"key": GOOD_KEY}
        assert GOOD_KEY not in tokens["access_token"]

    def test_state_and_issuer_come_back(self, app):
        registered = _register(app)
        pending = _authorize(app, registered["client_id"], state="xyz")
        location = _submit_key(app, pending, GOOD_KEY).headers["location"]
        assert location.startswith(REDIRECT + "?")
        assert _query(location)["state"] == "xyz"
        assert _query(location)["iss"] == PUBLIC  # RFC 9207
        assert app.get("/.well-known/oauth-authorization-server").json()[
            "authorization_response_iss_parameter_supported"
        ]

    def test_consent_page_names_the_app_and_offers_the_key_form(self, app):
        registered = _register(app)
        pending = _authorize(app, registered["client_id"])
        page = app.get("/oauth/consent", params={"request": pending})
        assert page.status_code == 200
        assert "<strong>claude.ai</strong>" in page.text
        assert 'name="api_key"' in page.text
        assert "Continue with Neuralk" not in page.text  # no Keycloak client configured
        assert page.headers["x-frame-options"] == "DENY"
        assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
        assert page.headers["referrer-policy"] == "no-referrer"

    def test_client_name_is_escaped(self, app):
        registered = _register(app, client_name="<script>x</script>")
        pending = _authorize(app, registered["client_id"])
        page = app.get("/oauth/consent", params={"request": pending})
        assert "<script>x</script>" not in page.text
        assert "&lt;script&gt;" in page.text

    def test_client_name_loses_bidi_overrides(self, app):
        registered = _register(app, client_name="Claude\u202eai.evil")
        assert registered["client_name"] == "Claudeai.evil"

    def test_private_scheme_redirect_is_shown_in_full(self, app):
        registered = _register(app, redirect_uris=["com.example.app:/oauth2redirect"])
        pending = _authorize(app, registered["client_id"], redirect_uri="com.example.app:/oauth2redirect")
        page = app.get("/oauth/consent", params={"request": pending}).text
        assert "the com.example.app app on this computer" in page
        assert "<code>com.example.app:/oauth2redirect</code>" in page

    def test_rejected_key_shows_the_reason(self, app):
        registered = _register(app)
        pending = _authorize(app, registered["client_id"])
        response = _submit_key(app, pending, "nk_live_bad")
        assert response.status_code == 400
        assert "rejected" in response.text
        assert "code=" not in response.headers.get("location", "")

    def test_empty_key(self, app):
        registered = _register(app)
        pending = _authorize(app, registered["client_id"])
        response = _submit_key(app, pending, "  ")
        assert response.status_code == 400
        assert "Paste your Neuralk API key" in response.text

    def test_expired_or_forged_request(self, app):
        assert app.get("/oauth/consent", params={"request": "seldon_req_forged"}).status_code == 400
        assert app.post("/oauth/consent", data={"request": "", "api_key": GOOD_KEY}).status_code == 400

    def test_form_posted_from_another_site_is_refused(self, app):
        # An auto-submitted cross-site form carries no sign-in cookie.
        registered = _register(app)
        pending = _authorize(app, registered["client_id"])
        response = app.post(
            "/oauth/consent",
            data={"request": pending, "api_key": GOOD_KEY, "action": "key", "csrf": "guess"},
            follow_redirects=False,
        )
        assert response.status_code == 400
        assert "location" not in response.headers

    def test_csrf_must_match_the_cookie(self, app):
        registered = _register(app)
        pending = _authorize(app, registered["client_id"])
        _open_page(app, pending)
        response = app.post(
            "/oauth/consent",
            data={"request": pending, "api_key": GOOD_KEY, "action": "key", "csrf": "not-the-cookie"},
            follow_redirects=False,
        )
        assert response.status_code == 400

    def test_sign_in_cookie_is_host_only_and_protected(self, app):
        registered = _register(app)
        pending = _authorize(app, registered["client_id"])
        page = app.get("/oauth/consent", params={"request": pending})
        cookie = page.headers["set-cookie"]
        # __Host-: no sibling subdomain can plant a value the check would accept.
        assert cookie.startswith("__Host-seldon_signin=")
        for attribute in ("HttpOnly", "Path=/;", "SameSite=lax", "Secure"):
            assert attribute.lower() in cookie.lower(), attribute
        assert "domain=" not in cookie.lower()

    def test_two_tabs_share_one_nonce(self, app):
        registered = _register(app)
        first = _authorize(app, registered["client_id"], state="a")
        second = _authorize(app, registered["client_id"], state="b")
        csrf_first = _open_page(app, first)
        csrf_second = _open_page(app, second)
        assert csrf_first == csrf_second
        response = app.post(
            "/oauth/consent",
            data={"request": first, "api_key": GOOD_KEY, "action": "key", "csrf": csrf_first},
            follow_redirects=False,
        )
        assert response.status_code == 302

    def test_non_ascii_csrf_is_a_400_not_a_crash(self, app):
        registered = _register(app)
        pending = _authorize(app, registered["client_id"])
        _open_page(app, pending)
        response = app.post(
            "/oauth/consent", data={"request": pending, "api_key": GOOD_KEY, "csrf": "é"}, follow_redirects=False
        )
        assert response.status_code == 400

    def test_buttons_are_armed_by_our_script_only(self, app):
        registered = _register(app)
        pending = _authorize(app, registered["client_id"])
        page = app.get("/oauth/consent", params={"request": pending})
        nonce = re.search(r"script-src 'nonce-([^']+)'", page.headers["content-security-policy"]).group(1)
        assert f'<script nonce="{nonce}">' in page.text

    def test_cancel_link_returns_access_denied(self, app):
        registered = _register(app)
        pending = _authorize(app, registered["client_id"], state="s9")
        page = app.get("/oauth/consent", params={"request": pending}).text
        assert "error=access_denied" in page
        assert "state=s9" in page


# --- the token endpoint ---


class TestTokenEndpoint:
    def _code(self, app) -> tuple[str, str]:
        registered = _register(app)
        pending = _authorize(app, registered["client_id"])
        code = _query(_submit_key(app, pending, GOOD_KEY).headers["location"])["code"]
        return registered["client_id"], code

    def test_code_is_single_use(self, app):
        client_id, code = self._code(app)
        assert _exchange(app, client_id, code).status_code == 200
        second = _exchange(app, client_id, code)
        assert second.status_code == 400
        assert second.json()["error"] == "invalid_grant"

    def test_wrong_verifier(self, app):
        client_id, code = self._code(app)
        response = _exchange(app, client_id, code, code_verifier="w" * 64)
        assert response.json()["error"] == "invalid_grant"

    def test_code_bound_to_its_client(self, app):
        _, code = self._code(app)
        other = _register(app)["client_id"]
        assert _exchange(app, other, code).json()["error"] == "invalid_grant"

    def test_redirect_uri_must_match(self, app):
        client_id, code = self._code(app)
        assert _exchange(app, client_id, code, redirect_uri="https://claude.ai/other").status_code == 400

    def test_confidential_client_needs_its_secret(self, app):
        registered = _register(app, token_endpoint_auth_method="client_secret_post")
        pending = _authorize(app, registered["client_id"])
        code = _query(_submit_key(app, pending, GOOD_KEY).headers["location"])["code"]
        assert _exchange(app, registered["client_id"], code, client_secret="wrong").status_code == 401
        good = _exchange(app, registered["client_id"], code, client_secret=registered["client_secret"])
        assert good.status_code == 200

    def test_refresh_rotates_both_tokens(self, app):
        registered, tokens = _sign_in_with_key(app)
        refreshed = app.post(
            "/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": tokens["refresh_token"],
                "client_id": registered["client_id"],
            },
        )
        assert refreshed.status_code == 200, refreshed.text
        new = refreshed.json()
        assert new["access_token"] != tokens["access_token"]
        assert new["refresh_token"] != tokens["refresh_token"]
        response = app.post("/mcp", json=INIT, headers={"Authorization": f"Bearer {new['access_token']}"})
        assert response.json() == {"key": GOOD_KEY}

    def test_refresh_after_the_key_was_revoked_asks_to_sign_in_again(self, app, neuralk):
        registered, tokens = _sign_in_with_key(app)
        neuralk.valid_keys.discard(GOOD_KEY)
        auth_mod.reset_cache()
        response = app.post(
            "/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": tokens["refresh_token"],
                "client_id": registered["client_id"],
            },
        )
        assert response.status_code == 400
        assert response.json()["error"] == "invalid_grant"

    def _refresh(self, app, client_id, refresh_token):
        return app.post(
            "/token", data={"grant_type": "refresh_token", "refresh_token": refresh_token, "client_id": client_id}
        )

    def test_refresh_token_is_single_use_after_a_grace(self, app, monkeypatch):
        registered, tokens = _sign_in_with_key(app)
        assert self._refresh(app, registered["client_id"], tokens["refresh_token"]).status_code == 200
        # A retry right away (the client lost the answer) still works...
        assert self._refresh(app, registered["client_id"], tokens["refresh_token"]).status_code == 200
        # ...a replay later does not.
        now = time.time()
        monkeypatch.setattr(oauth_mod.time, "time", lambda: now + oauth_mod.REFRESH_REUSE_GRACE_S + 1)
        replay = self._refresh(app, registered["client_id"], tokens["refresh_token"])
        assert replay.status_code == 400
        assert replay.json()["error"] == "invalid_grant"

    def test_sign_in_ends_after_the_session_lifetime(self, app, monkeypatch):
        registered, tokens = _sign_in_with_key(app)
        now = time.time()
        # Refreshing keeps it going, but never past the session end.
        monkeypatch.setattr(oauth_mod.time, "time", lambda: now + oauth_mod.SESSION_TTL_S - 100)
        last = self._refresh(app, registered["client_id"], tokens["refresh_token"]).json()
        assert last["expires_in"] <= 100
        monkeypatch.setattr(oauth_mod.time, "time", lambda: now + oauth_mod.SESSION_TTL_S + 1)
        assert self._refresh(app, registered["client_id"], last["refresh_token"]).json()["error"] == "invalid_grant"

    def test_access_token_is_not_a_refresh_token(self, app):
        registered, tokens = _sign_in_with_key(app)
        response = app.post(
            "/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": tokens["access_token"],
                "client_id": registered["client_id"],
            },
        )
        assert response.json()["error"] == "invalid_grant"


# --- the MCP endpoint with tokens ---


class TestAccessTokens:
    def test_expired_token_gets_invalid_token(self, app, monkeypatch):
        _, tokens = _sign_in_with_key(app)
        now = time.time()
        monkeypatch.setattr(oauth_mod.time, "time", lambda: now + oauth_mod.ACCESS_TOKEN_TTL_S + 1)
        response = app.post("/mcp", json=INIT, headers={"Authorization": f"Bearer {tokens['access_token']}"})
        assert response.status_code == 401
        assert 'error="invalid_token"' in response.headers["www-authenticate"]
        assert "resource_metadata=" in response.headers["www-authenticate"]

    def test_tampered_token_gets_invalid_token(self, app):
        _, tokens = _sign_in_with_key(app)
        token = tokens["access_token"]
        tampered = token[:-4] + ("AAAA" if not token.endswith("AAAA") else "BBBB")
        response = app.post("/mcp", json=INIT, headers={"Authorization": f"Bearer {tampered}"})
        assert response.status_code == 401
        assert 'error="invalid_token"' in response.headers["www-authenticate"]

    def test_token_of_another_server_secret_is_rejected(self, app, monkeypatch):
        other = Sealer("y" * 48).seal("access", {"c": "c", "k": GOOD_KEY}, 3600)
        response = app.post("/mcp", json=INIT, headers={"Authorization": f"Bearer {other}"})
        assert response.status_code == 401

    def test_refresh_token_is_not_a_credential(self, app):
        # Never passed on to Neuralk as if it were an API key.
        _, tokens = _sign_in_with_key(app)
        response = app.post("/mcp", json=INIT, headers={"Authorization": f"Bearer {tokens['refresh_token']}"})
        assert response.status_code == 401
        assert 'error="invalid_token"' in response.headers["www-authenticate"]

    def test_access_token_in_a_key_header_is_opened(self, app):
        _, tokens = _sign_in_with_key(app)
        response = app.post("/mcp", json=INIT, headers={"x-api-key": tokens["access_token"]})
        assert response.json() == {"key": GOOD_KEY}

    def test_revoked_key_makes_the_client_sign_in_again(self, app, neuralk):
        _, tokens = _sign_in_with_key(app)
        neuralk.valid_keys.discard(GOOD_KEY)
        auth_mod.reset_cache()
        response = app.post("/mcp", json=INIT, headers={"Authorization": f"Bearer {tokens['access_token']}"})
        assert response.status_code == 401
        assert 'error="invalid_token"' in response.headers["www-authenticate"]


# --- "Continue with Neuralk" ---


class TestNeuralkSignIn:
    def _to_keycloak(self, client: TestClient) -> tuple[dict, str, dict]:
        registered = _register(client)
        pending = _authorize(client, registered["client_id"], state="st-kc")
        assert "Continue with Neuralk" in client.get("/oauth/consent", params={"request": pending}).text
        csrf = _open_page(client, pending)
        response = client.post(
            "/oauth/consent", data={"request": pending, "action": "neuralk", "csrf": csrf}, follow_redirects=False
        )
        assert response.status_code == 303, response.text
        location = response.headers["location"]
        assert location.startswith(f"{KC}/protocol/openid-connect/auth?")
        return registered, pending, _query(location)

    def test_full_flow_creates_a_key_for_the_connection(self, app_with_neuralk_sign_in, neuralk):
        client = app_with_neuralk_sign_in
        registered, _, kc = self._to_keycloak(client)
        assert kc["client_id"] == "seldon-mcp"
        assert kc["redirect_uri"] == f"{PUBLIC}/oauth/callback"
        assert kc["code_challenge_method"] == "S256"

        back = client.get("/oauth/callback", params={"code": "kc-code", "state": kc["state"]}, follow_redirects=False)
        assert back.status_code == 302, back.text
        answer = _query(back.headers["location"])
        assert answer["state"] == "st-kc"

        # Keycloak got our client credentials and the PKCE verifier matching the challenge.
        form = neuralk.kc_token_forms[0]
        assert form["client_secret"] == "kc-secret"
        expected = base64.urlsafe_b64encode(hashlib.sha256(form["code_verifier"].encode()).digest()).decode()
        assert expected.rstrip("=") == kc["code_challenge"]
        # The key was created in the user's organization, able to upload and
        # predict, and expires when the sign-in does.
        assert neuralk.created[0]["scopes"] == ["read", "write"]
        assert neuralk.created[0]["name"].startswith("Claude connector (")
        expires = datetime.fromisoformat(neuralk.created[0]["expires_at"]).timestamp()
        assert abs(expires - (time.time() + oauth_mod.SESSION_TTL_S)) < 60

        tokens = _exchange(client, registered["client_id"], answer["code"]).json()
        response = client.post("/mcp", json=INIT, headers={"Authorization": f"Bearer {tokens['access_token']}"})
        assert response.json() == {"key": "nk_live_created"}

    def test_member_without_key_rights_is_told_to_paste_a_key(self, app_with_neuralk_sign_in, neuralk):
        neuralk.create_status = 403
        neuralk.create_body = {
            "detail": {"error": {"code": 4030104, "message": "Insufficient role for key management"}}
        }
        client = app_with_neuralk_sign_in
        _, _, kc = self._to_keycloak(client)
        page = client.get("/oauth/callback", params={"code": "kc-code", "state": kc["state"]})
        assert page.status_code == 400
        assert "ask an admin or owner" in page.text
        assert 'name="api_key"' in page.text  # the key form is right there

    def test_expired_trial_message_is_shown(self, app_with_neuralk_sign_in, neuralk):
        neuralk.create_status = 403
        neuralk.create_body = {
            "detail": {
                "error": {"code": 4030110, "message": "Organization trial has expired. Please upgrade to continue."}
            }
        }
        client = app_with_neuralk_sign_in
        _, _, kc = self._to_keycloak(client)
        page = client.get("/oauth/callback", params={"code": "kc-code", "state": kc["state"]})
        assert "trial has expired" in page.text

    def test_cancelled_at_keycloak(self, app_with_neuralk_sign_in):
        client = app_with_neuralk_sign_in
        _, _, kc = self._to_keycloak(client)
        page = client.get("/oauth/callback", params={"error": "access_denied", "state": kc["state"]})
        assert page.status_code == 400
        assert "cancelled" in page.text

    def test_callback_in_another_browser_mints_nothing(self, app_with_neuralk_sign_in, neuralk):
        # An attacker starts a sign-in in their browser and sends the Keycloak
        # link to a victim already logged in to Neuralk: Keycloak sends the
        # victim back straight away, but the victim's browser has no sign-in cookie.
        client = app_with_neuralk_sign_in
        _, _, kc = self._to_keycloak(client)
        client.cookies.clear()  # the victim's browser
        page = client.get(
            "/oauth/callback", params={"code": "kc-code", "state": kc["state"]}, follow_redirects=False
        )
        assert page.status_code == 400
        assert "did not start here" in page.text
        assert neuralk.created == []
        assert neuralk.kc_token_forms == []

    def test_forged_state(self, app_with_neuralk_sign_in):
        page = app_with_neuralk_sign_in.get("/oauth/callback", params={"code": "c", "state": "seldon_up_forged"})
        assert page.status_code == 400
        assert "expired" in page.text

    def test_button_ignored_without_a_keycloak_client(self, app):
        registered = _register(app)
        pending = _authorize(app, registered["client_id"])
        csrf = _open_page(app, pending)
        response = app.post(
            "/oauth/consent", data={"request": pending, "action": "neuralk", "csrf": csrf}, follow_redirects=False
        )
        assert response.status_code == 400  # treated as the key form, with no key
        assert "Paste your Neuralk API key" in response.text


# --- "Continue with Neuralk" through the dashboard ---

DASHBOARD = "https://dash.test"


@pytest.fixture
def app_with_dashboard(monkeypatch, neuralk):
    client, _ = _make_app(
        _config(
            neuralk_dashboard_url=DASHBOARD + "/",
            # The dashboard wins over a Keycloak client configured alongside.
            neuralk_oidc_client_id="seldon-mcp",
            neuralk_oidc_client_secret="kc-secret",
        ),
        monkeypatch,
    )
    return client


class TestDashboardSignIn:
    def _to_dashboard(self, client: TestClient, state: str = "st-d") -> tuple[dict, str]:
        registered = _register(client)
        pending = _authorize(client, registered["client_id"], state=state)
        page = client.get("/oauth/consent", params={"request": pending}).text
        assert "password or a magic link" in page
        csrf = _open_page(client, pending)
        response = client.post(
            "/oauth/consent", data={"request": pending, "action": "neuralk", "csrf": csrf}, follow_redirects=False
        )
        assert response.status_code == 303, response.text
        location = response.headers["location"]
        assert location.startswith(f"{DASHBOARD}/connect?state=seldon_up_")
        return registered, _query(location)["state"]

    def _describe(self, client: TestClient, state: str, origin: str = DASHBOARD):
        return client.get("/oauth/connect/describe", params={"state": state}, headers={"Origin": origin})

    def test_describe_tells_the_dashboard_what_to_show(self, app_with_dashboard):
        client = app_with_dashboard
        _, state = self._to_dashboard(client)
        response = self._describe(client, state)
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == DASHBOARD
        assert response.headers["access-control-allow-credentials"] == "true"
        body = response.json()
        assert body["app"] == "claude.ai"
        assert body["client_name"] == "Claude"
        assert body["destination"] is None
        assert body["key"]["scopes"] == ["read", "write"]
        assert body["key"]["name"].startswith("Claude connector (")
        expires = datetime.fromisoformat(body["key"]["expires_at"]).timestamp()
        assert abs(expires - (time.time() + oauth_mod.SESSION_TTL_S)) < 60
        assert body["back_url"].startswith(f"{PUBLIC}/oauth/consent?request=seldon_req_")

    def test_describe_for_another_origin_is_not_readable(self, app_with_dashboard):
        client = app_with_dashboard
        _, state = self._to_dashboard(client)
        response = self._describe(client, state, origin="https://evil.example")
        assert "access-control-allow-origin" not in response.headers

    def test_describe_in_another_browser_is_refused(self, app_with_dashboard):
        # A dashboard link built from someone else's sign-in: nothing to show,
        # so the dashboard creates no key.
        client = app_with_dashboard
        _, state = self._to_dashboard(client)
        client.cookies.clear()
        response = self._describe(client, state)
        assert response.status_code == 400
        assert "did not start in this browser" in response.json()["error"]

    def test_complete_with_the_created_key(self, app_with_dashboard, neuralk):
        client = app_with_dashboard
        registered, state = self._to_dashboard(client)
        neuralk.valid_keys.add("nk_live_from_dashboard")
        back = client.post(
            "/oauth/connect/complete",
            data={"state": state, "api_key": "nk_live_from_dashboard"},
            follow_redirects=False,
        )
        assert back.status_code == 302, back.text
        answer = _query(back.headers["location"])
        assert back.headers["location"].startswith(REDIRECT + "?")
        assert answer["state"] == "st-d"
        tokens = _exchange(client, registered["client_id"], answer["code"]).json()
        response = client.post("/mcp", json=INIT, headers={"Authorization": f"Bearer {tokens['access_token']}"})
        assert response.json() == {"key": "nk_live_from_dashboard"}

    def test_complete_with_a_rejected_key(self, app_with_dashboard):
        client = app_with_dashboard
        _, state = self._to_dashboard(client)
        response = client.post(
            "/oauth/connect/complete", data={"state": state, "api_key": "nk_live_bad"}, follow_redirects=False
        )
        assert response.status_code == 400
        assert "rejected" in response.text
        assert 'name="api_key"' in response.text  # back on the sign-in page

    def test_cancel_returns_access_denied(self, app_with_dashboard):
        client = app_with_dashboard
        _, state = self._to_dashboard(client)
        back = client.post(
            "/oauth/connect/complete", data={"state": state, "action": "cancel"}, follow_redirects=False
        )
        assert back.status_code == 302
        assert _query(back.headers["location"])["error"] == "access_denied"

    def test_complete_in_another_browser_is_refused(self, app_with_dashboard):
        client = app_with_dashboard
        _, state = self._to_dashboard(client)
        client.cookies.clear()
        response = client.post(
            "/oauth/connect/complete", data={"state": state, "api_key": GOOD_KEY}, follow_redirects=False
        )
        assert response.status_code == 400
        assert "location" not in response.headers

    def test_a_keycloak_state_is_not_a_dashboard_state(self, app_with_dashboard):
        client = app_with_dashboard
        registered = _register(client)
        pending = _authorize(client, registered["client_id"])
        _open_page(client, pending)
        nonce = client.cookies.get("__Host-seldon_signin")
        sealer = Sealer(SECRET)
        keycloak_state = sealer.seal("upstream", {"q": pending, "v": "x", "n": nonce}, 600)
        assert self._describe(client, keycloak_state).status_code == 400

    def test_endpoints_off_without_a_dashboard(self, app):
        assert app.get("/oauth/connect/describe", params={"state": "x"}).status_code == 404
        assert app.post("/oauth/connect/complete", data={"state": "x"}).status_code == 400


def test_keycloak_mode_warns_magic_link_accounts(app_with_neuralk_sign_in):
    client = app_with_neuralk_sign_in
    registered = _register(client)
    pending = _authorize(client, registered["client_id"])
    page = client.get("/oauth/consent", params={"request": pending}).text
    assert "Only sign in with magic links? Use an API key below." in page


# --- building blocks ---


class TestSealer:
    def test_round_trip_and_kinds(self):
        sealer = Sealer(SECRET)
        token = sealer.seal("access", {"k": "v"}, 60)
        assert sealer.open("access", token)["k"] == "v"
        assert sealer.open("refresh", token) is None
        assert sealer.open("refresh", "seldon_rt_" + token.removeprefix("seldon_at_")) is None

    def test_expiry(self, monkeypatch):
        sealer = Sealer(SECRET)
        token = sealer.seal("code", {"k": "v"}, 10)
        now = time.time()
        monkeypatch.setattr(oauth_mod.time, "time", lambda: now + 11)
        assert sealer.open("code", token) is None

    def test_garbage(self):
        sealer = Sealer(SECRET)
        for token in (None, "", "seldon_at_", "seldon_at_!!!", "seldon_at_" + "A" * 80, "nk_live_x"):
            assert sealer.open("access", token) is None

    def test_short_secret_refused(self):
        with pytest.raises(ValueError, match="at least"):
            Sealer("short")


class TestRedirectMatching:
    def test_exact(self):
        assert redirect_uri_matches(REDIRECT, REDIRECT)
        assert not redirect_uri_matches(REDIRECT, "https://claude.ai/api/mcp/other")

    def test_loopback_any_port(self):
        assert redirect_uri_matches("http://localhost:1111/callback", "http://localhost:2222/callback")
        assert redirect_uri_matches("http://127.0.0.1/callback", "http://127.0.0.1:5555/callback")
        assert not redirect_uri_matches("http://localhost:1111/callback", "http://127.0.0.1:1111/callback")
        assert not redirect_uri_matches("http://localhost:1111/callback", "http://localhost:1111/other")
        assert not redirect_uri_matches("https://claude.ai:443/cb", "https://claude.ai:8443/cb")

    def test_claude_code_reuses_its_client_on_a_new_port(self, app):
        registered = _register(app, redirect_uris=["http://localhost:3118/callback"])
        pending = _authorize(app, registered["client_id"], redirect_uri="http://localhost:4242/callback")
        location = _submit_key(app, pending, GOOD_KEY).headers["location"]
        assert location.startswith("http://localhost:4242/callback?")


class TestConfiguration:
    def test_needs_a_public_url(self):
        with pytest.raises(ValueError, match="SELDON_PUBLIC_URL"):
            OAuthServer(SeldonConfig(seldon_oauth_secret=SECRET))

    def test_public_url_must_be_an_origin(self):
        with pytest.raises(ValueError, match="bare origin"):
            OAuthServer(SeldonConfig(seldon_oauth_secret=SECRET, seldon_public_url="https://host/prefix"))
        OAuthServer(SeldonConfig(seldon_oauth_secret=SECRET, seldon_public_url="https://host/"))

    def test_access_log_keeps_sign_in_queries_out(self):
        import logging

        record = logging.LogRecord(
            "uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
            ("1.2.3.4:5", "GET", "/oauth/callback?code=secret&state=s", "1.1", 302), None,
        )
        server._RedactSignInQueries().filter(record)
        assert "secret" not in record.getMessage()
        assert "/oauth/callback?…" in record.getMessage()
        other = logging.LogRecord(
            "uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
            ("1.2.3.4:5", "GET", "/healthz?x=1", "1.1", 200), None,
        )
        server._RedactSignInQueries().filter(other)
        assert "/healthz?x=1" in other.getMessage()

    def test_built_app_serves_oauth_and_gates_everything(self, monkeypatch, tmp_path):
        monkeypatch.setattr(server, "_lifespan_config", None)
        monkeypatch.setattr(server, "_download_store", None)
        monkeypatch.setattr(server, "_oauth", None)
        _init_state(_config(seldon_download_dir=str(tmp_path)))
        client = TestClient(build_http_app("0.0.0.0", 8000))
        assert client.get("/healthz").text == "ok"
        assert client.get("/.well-known/oauth-protected-resource/mcp").status_code == 200
        response = client.post("/mcp", json=INIT)
        assert response.status_code == 401
        assert "resource_metadata=" in response.headers["www-authenticate"]
        assert client.get("/downloads/nope").status_code == 404
        assert server._oauth is not None

    def test_without_secret_discovery_stays_open(self, monkeypatch, tmp_path):
        monkeypatch.setattr(server, "_lifespan_config", None)
        monkeypatch.setattr(server, "_download_store", None)
        _init_state(SeldonConfig(require_client_api_key=True, seldon_download_dir=str(tmp_path)))
        client = TestClient(build_http_app("0.0.0.0", 8000))
        assert server._oauth is None
        assert client.get("/.well-known/oauth-protected-resource/mcp").status_code == 404
        call = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "list_models"}}
        assert client.post("/mcp", json=call).status_code == 401
