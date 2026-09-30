"""OAuth sign-in for the hosted server.

A Claude user adds ``https://mcp.neuralk.ai/mcp`` as a connector and presses
Connect; ChatGPT, Claude Code, Cursor and every MCP client that speaks the MCP
authorization spec do the same. The server answers as its own OAuth 2.1
authorization server: the client registers itself (RFC 7591), sends the user
to a Neuralk page, and gets tokens back. On that page the user either
continues with their Neuralk account, or pastes a key they already have.
"Continue with Neuralk" goes to the Neuralk dashboard's /connect page when
NEURALK_DASHBOARD_URL is set: the user signs in there however they usually do
(password or magic link), approves, and the dashboard creates an API key for
the connection and posts it back here. Otherwise it signs in on the Keycloak
realm, and this server creates the key with the user's token. Either way an
API key is what the prediction API takes, and the tokens the client holds are
that key, sealed.

Stateless where it can be. The client registration, the pending authorization,
the code and both tokens are AES-GCM blobs under ``SELDON_OAUTH_SECRET``: there
is no database, and a redeploy signs nobody out. What needs memory is kept per
process, which the chart's single replica makes enough: the codes already
exchanged and the refresh tokens already rotated, both single-use (a refresh
token gets a minute of grace for a retried request). A sign-in lasts 90 days,
after which the user presses Connect again; a key created at sign-in expires
with it. Revoking the API key in the dashboard ends a connection sooner (keys
are checked again every few minutes), and rotating the secret ends them all.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import html
import json
import logging
import os
import re
import secrets
import time
import unicodedata
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode, urlsplit

import httpx
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from mcp.server.auth.handlers.authorize import AuthorizationHandler
from mcp.server.auth.handlers.token import TokenHandler
from mcp.server.auth.middleware.client_auth import ClientAuthenticator
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.server.auth.routes import cors_middleware
from mcp.shared.auth import InvalidRedirectUriError, OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from pydantic import AnyUrl, ValidationError
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route

from seldon_mcp.auth import APIKeyAuthError, _SkipValidation, validate_api_key
from seldon_mcp.config import SeldonConfig

logger = logging.getLogger("seldon_mcp.oauth")

# The one scope there is: run Seldon on the user's Neuralk account.
SCOPE = "seldon"
ACCESS_TOKEN_TTL_S = 3600
# How long a sign-in lasts. Refresh tokens end with it, and so does a key
# created at sign-in: then the user presses Connect again.
SESSION_TTL_S = 90 * 24 * 3600
CODE_TTL_S = 300
# How long the user has on the sign-in page (and at the Neuralk login).
PENDING_TTL_S = 15 * 60
# A refresh token is single-use, but a client that lost the answer to a
# refresh (a network error) retries with the same one: it gets this long.
REFRESH_REUSE_GRACE_S = 60
# How long a used refresh token is remembered, which bounds the memory it
# takes. A token replayed later than this is not caught.
SPENT_REFRESH_RETENTION_S = 7 * 24 * 3600
MIN_SECRET_LENGTH = 32

CONSENT_PATH = "/oauth/consent"
CALLBACK_PATH = "/oauth/callback"
# The dashboard's side of "Continue with Neuralk": its /connect page asks
# what to show (describe), then posts the key it created (complete).
DASHBOARD_CONNECT_PATH = "/connect"
CONNECT_DESCRIBE_PATH = "/oauth/connect/describe"
CONNECT_COMPLETE_PATH = "/oauth/connect/complete"
API_KEYS_URL = "https://prediction.neuralk-ai.com/dashboard/api-keys"
DOCS_URL = "https://docs.neuralk.ai/integrations/mcp.html"
# What a key created at sign-in may do: `read` runs inference, `write` uploads.
CREATED_KEY_SCOPES = ["read", "write"]

# Every blob starts with this, then its kind. Each kind is sealed with the
# kind as associated data, so one never opens as another: a code is not a
# token, a refresh token is not an access token.
TOKEN_PREFIX = "seldon_"
_PREFIXES = {
    "client": "seldon_client_",
    "pending": "seldon_req_",
    "upstream": "seldon_up_",
    "code": "seldon_code_",
    "access": "seldon_at_",
    "refresh": "seldon_rt_",
}
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
# Private-use redirect schemes of MCP clients that predate RFC 8252's
# reverse-DNS naming (section 7.1), which any other native app must use.
_KNOWN_APP_SCHEMES = {"cursor", "vscode", "vscode-insiders", "windsurf", "zed"}
_NONCE = re.compile(r"[A-Za-z0-9_-]{16,64}")


class Sealer:
    """Authenticated encryption of small JSON payloads under the server secret.

    AES-256-GCM with a random 96-bit nonce, the key derived from the secret
    with HKDF (so the secret can be any long random string). A blob carries
    its own expiry, checked on opening.
    """

    _VERSION = b"\x01"

    def __init__(self, secret: str) -> None:
        if len(secret) < MIN_SECRET_LENGTH:
            raise ValueError(f"SELDON_OAUTH_SECRET must be at least {MIN_SECRET_LENGTH} characters")
        material = secret.encode()
        self._aead = AESGCM(_hkdf(material, b"seldon-mcp oauth seal v1"))
        self._mac_key = _hkdf(material, b"seldon-mcp oauth client secret v1")

    def seal(self, kind: str, payload: dict[str, Any], ttl_s: int | None) -> str:
        body = dict(payload)
        if ttl_s is not None:
            body["exp"] = int(time.time()) + ttl_s
        nonce = os.urandom(12)
        sealed = self._aead.encrypt(nonce, json.dumps(body, separators=(",", ":")).encode(), kind.encode())
        return _PREFIXES[kind] + base64.urlsafe_b64encode(self._VERSION + nonce + sealed).decode().rstrip("=")

    def open(self, kind: str, token: str | None) -> dict[str, Any] | None:
        """The payload of a blob of this kind, or None if forged, of another kind, or expired."""
        prefix = _PREFIXES[kind]
        if not token or not token.startswith(prefix):
            return None
        raw = token[len(prefix):]
        try:
            blob = base64.b64decode(raw + "=" * (-len(raw) % 4), altchars=b"-_", validate=True)
        except (ValueError, binascii.Error):
            return None
        if len(blob) < 1 + 12 + 16 or blob[:1] != self._VERSION:
            return None
        try:
            payload = json.loads(self._aead.decrypt(blob[1:13], blob[13:], kind.encode()))
        except (InvalidTag, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        expires = payload.get("exp")
        if expires is not None and expires < time.time():
            return None
        return payload

    def client_secret(self, client_id: str) -> str:
        """The secret of a confidential client, derived from its id: nothing to store."""
        return hmac.new(self._mac_key, client_id.encode(), hashlib.sha256).hexdigest()


def _hkdf(material: bytes, info: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(material)


def _digest(client_id: str) -> str:
    """A short, stable stand-in for a (long) client id inside codes and tokens."""
    return hashlib.sha256(client_id.encode()).hexdigest()[:32]


def _same(a: str, b: str) -> bool:
    """Constant-time equality of two strings, whatever characters they hold."""
    return hmac.compare_digest(a.encode(), b.encode())


def _clean_name(name: str | None) -> str | None:
    """A client's self-chosen name, without control or formatting characters (bidi overrides)."""
    if not name:
        return None
    cleaned = "".join(ch for ch in name if not unicodedata.category(ch).startswith("C")).strip()
    return cleaned[:100] or None


def redirect_uri_matches(registered: str, requested: str) -> bool:
    """Exact match, except that a loopback redirect matches on any port.

    A native client (Claude Code) binds a fresh port for every sign-in and
    reuses the client it registered once, so the port it asks for is never
    the one it registered (RFC 8252 section 7.3).
    """
    if registered == requested:
        return True
    a, b = urlsplit(registered), urlsplit(requested)
    return (
        a.scheme == b.scheme == "http"
        and a.hostname in _LOOPBACK_HOSTS
        and a.hostname == b.hostname
        and a.path == b.path
        and a.query == b.query
    )


def _redirect_uri_problem(uri: str) -> str | None:
    """Why a client may not register this redirect URI, or None if it may.

    An allowlist: https anywhere, http on loopback, and a native app's
    private-use scheme. Not a scheme that opens a web page, such as
    x-safari-https or googlechromes, which would hand the code to a site
    while the sign-in page says "an app on this computer".
    """
    parts = urlsplit(uri)
    scheme = parts.scheme.lower()
    if parts.fragment:
        return f"{uri}: a redirect URI must not have a fragment"
    if scheme == "https":
        return None if parts.hostname else f"{uri}: no host"
    if scheme == "http":
        return None if parts.hostname in _LOOPBACK_HOSTS else f"{uri}: plain http is for loopback redirects only"
    private = scheme in _KNOWN_APP_SCHEMES or ("." in scheme and "http" not in scheme)
    if private and not re.match(r"/*https?:", uri[len(scheme) + 1:], re.IGNORECASE):
        return None
    return f"{uri}: use https, http on a loopback address, or an app's reverse-DNS scheme (RFC 8252)"


def _app_label(redirect_uri: str) -> str:
    """Who is asking, as the user can check it: where the answer goes."""
    parts = urlsplit(redirect_uri)
    if parts.scheme == "https" and parts.hostname:
        return parts.hostname
    if parts.scheme == "http" and parts.hostname in _LOOPBACK_HOSTS:
        return "an app on this computer"
    return f"the {parts.scheme} app on this computer"


class RegisteredClient(OAuthClientInformationFull):
    """A registered client, rebuilt from its sealed id."""

    def validate_redirect_uri(self, redirect_uri: AnyUrl | None) -> AnyUrl:
        if redirect_uri is None:
            return super().validate_redirect_uri(None)
        if any(redirect_uri_matches(str(uri), str(redirect_uri)) for uri in self.redirect_uris or ()):
            return redirect_uri
        raise InvalidRedirectUriError(f"Redirect URI '{redirect_uri}' not registered for client")

    def validate_scope(self, requested_scope: str | None) -> list[str]:
        # One scope is granted, whatever was asked: unknown scopes (openid,
        # offline_access) are dropped rather than refused, which RFC 6749
        # section 3.3 allows and which keeps a client that asks for them working.
        return [SCOPE]


class SealedCode(AuthorizationCode):
    api_key: str
    jti: str
    session_end: int


class SealedRefreshToken(RefreshToken):
    api_key: str
    jti: str
    session_end: int
    resource: str | None = None


class SealedProvider:
    """The SDK's OAuthAuthorizationServerProvider, over sealed blobs.

    The SDK's authorize and token handlers do the protocol (parameter
    validation, PKCE, redirect URI checks, error shapes); this class only
    mints and opens what they ask for.
    """

    def __init__(self, sealer: Sealer, consent_url: str, config: SeldonConfig) -> None:
        self.sealer = sealer
        self.consent_url = consent_url
        self.config = config
        self._spent_codes: dict[str, float] = {}  # jti -> when the code expires anyway
        self._spent_refresh: dict[str, tuple[float, float]] = {}  # jti -> (spent at, forget at)

    def client(self, client_id: str) -> RegisteredClient | None:
        data = self.sealer.open("client", client_id)
        if data is None or not data.get("r"):
            return None
        method = data["m"]
        return RegisteredClient(
            client_id=client_id,
            client_secret=None if method == "none" else self.sealer.client_secret(client_id),
            client_id_issued_at=data.get("t"),
            redirect_uris=data["r"],
            token_endpoint_auth_method=method,
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            client_name=data.get("n"),
            scope=SCOPE,
        )

    async def get_client(self, client_id: str) -> RegisteredClient | None:
        return self.client(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        raise NotImplementedError("registration is stateless: OAuthServer.register mints the client id")

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        """Send the user to the sign-in page, what the rest of the flow needs sealed in its URL."""
        pending = self.sealer.seal(
            "pending",
            {
                "c": _digest(client.client_id or ""),
                "n": client.client_name,
                "r": str(params.redirect_uri),
                "x": params.redirect_uri_provided_explicitly,
                "s": params.state,
                "p": params.code_challenge,
                "rs": params.resource,
            },
            PENDING_TTL_S,
        )
        return f"{self.consent_url}?{urlencode({'request': pending})}"

    def issue_code(self, pending: dict[str, Any], api_key: str, session_end: int) -> str:
        return self.sealer.seal(
            "code",
            {
                "c": pending["c"],
                "r": pending["r"],
                "x": pending["x"],
                "p": pending["p"],
                "rs": pending.get("rs"),
                "k": api_key,
                "j": secrets.token_urlsafe(12),
                "se": session_end,
            },
            CODE_TTL_S,
        )

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> SealedCode | None:
        data = self.sealer.open("code", authorization_code)
        if data is None or data["c"] != _digest(client.client_id or "") or data["j"] in self._spent_codes:
            return None
        return SealedCode(
            code=authorization_code,
            scopes=[SCOPE],
            expires_at=data["exp"],
            client_id=client.client_id or "",
            code_challenge=data["p"],
            redirect_uri=AnyUrl(data["r"]),
            redirect_uri_provided_explicitly=data["x"],
            resource=data.get("rs"),
            api_key=data["k"],
            jti=data["j"],
            session_end=data["se"],
        )

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: SealedCode
    ) -> OAuthToken:
        # A code is single-use. Checked and spent with no await in between, so
        # two concurrent exchanges of one code cannot both get tokens.
        now = time.time()
        self._spent_codes = {jti: exp for jti, exp in self._spent_codes.items() if exp > now}
        if authorization_code.jti in self._spent_codes:
            raise TokenError(error="invalid_grant", error_description="authorization code already used")
        self._spent_codes[authorization_code.jti] = authorization_code.expires_at
        return self._tokens(
            client.client_id or "", authorization_code.api_key, authorization_code.resource,
            authorization_code.session_end,
        )

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> SealedRefreshToken | None:
        data = self.sealer.open("refresh", refresh_token)
        if data is None or data["c"] != _digest(client.client_id or ""):
            return None
        return SealedRefreshToken(
            token=refresh_token,
            client_id=client.client_id or "",
            scopes=[SCOPE],
            expires_at=data["exp"],
            api_key=data["k"],
            jti=data["j"],
            session_end=data["se"],
            resource=data.get("rs"),
        )

    async def exchange_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: SealedRefreshToken, scopes: list[str]
    ) -> OAuthToken:
        # Single-use (OAuth 2.1 section 4.3.1), spent before the await below so
        # two refreshes racing with one token are told apart.
        now = time.time()
        self._spent_refresh = {jti: v for jti, v in self._spent_refresh.items() if v[1] > now}
        spent = self._spent_refresh.get(refresh_token.jti)
        if spent is not None and now - spent[0] > REFRESH_REUSE_GRACE_S:
            raise TokenError(error="invalid_grant", error_description="refresh token already used")
        if spent is None:
            forget_at = min(float(refresh_token.expires_at or now), now + SPENT_REFRESH_RETENTION_S)
            self._spent_refresh[refresh_token.jti] = (now, forget_at)
        # A key revoked in the dashboard ends the connection here: the client
        # is told to sign in again instead of getting tokens that fail on use.
        if not self.config.skip_api_key_validation:
            try:
                await _validate(refresh_token.api_key, self.config)
            except APIKeyAuthError as exc:
                raise TokenError(error="invalid_grant", error_description=str(exc)) from exc
            except _SkipValidation:
                pass  # the auth API is down: keep the user connected
        return self._tokens(
            client.client_id or "", refresh_token.api_key, refresh_token.resource, refresh_token.session_end
        )

    async def load_access_token(self, token: str) -> AccessToken | None:
        data = self.sealer.open("access", token)
        if data is None:
            return None
        return AccessToken(
            token=token, client_id=data["c"], scopes=[SCOPE], expires_at=data["exp"], resource=data.get("rs")
        )

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        return None  # stateless: revoke the API key instead (module docstring)

    def _tokens(self, client_id: str, api_key: str, resource: str | None, session_end: int) -> OAuthToken:
        remaining = max(1, session_end - int(time.time()))
        access_ttl = min(ACCESS_TOKEN_TTL_S, remaining)
        claims = {"c": _digest(client_id), "k": api_key, "rs": resource, "se": session_end}
        return OAuthToken(
            access_token=self.sealer.seal("access", claims, access_ttl),
            token_type="Bearer",
            expires_in=access_ttl,
            scope=SCOPE,
            refresh_token=self.sealer.seal("refresh", {**claims, "j": secrets.token_urlsafe(12)}, remaining),
        )


async def _validate(api_key: str, config: SeldonConfig) -> None:
    await validate_api_key(
        api_key,
        base_url=config.neuralk_prediction_url,
        ttl_s=config.api_key_validation_ttl_s,
        timeout_s=config.api_key_validation_timeout_s,
    )


class SignInError(Exception):
    """A sign-in step failed; the message is shown to the user on the sign-in page."""


class OAuthServer:
    """The authorization server and the resource-side checks of the hosted server.

    Args:
        config: The server configuration. ``seldon_oauth_secret`` and
            ``seldon_public_url`` (a bare origin: it is the issuer) must be set.

    Raises:
        ValueError: If the secret is missing or short, or the public URL is
            missing or has a path.
    """

    def __init__(self, config: SeldonConfig) -> None:
        if not config.seldon_oauth_secret:
            raise ValueError("SELDON_OAUTH_SECRET is not set")
        if not config.public_url:
            raise ValueError("SELDON_PUBLIC_URL must be set with SELDON_OAUTH_SECRET: it is the OAuth issuer")
        parts = urlsplit(config.public_url)
        if parts.path.strip("/") or parts.query or parts.fragment:
            raise ValueError("SELDON_PUBLIC_URL must be a bare origin (https://host) with SELDON_OAUTH_SECRET")
        self.config = config
        self.issuer = config.public_url.rstrip("/")
        self.resource = f"{self.issuer}/mcp"
        self.sealer = Sealer(config.seldon_oauth_secret)
        self.provider = SealedProvider(self.sealer, f"{self.issuer}{CONSENT_PATH}", config)
        # "Continue with Neuralk": through the dashboard, else through
        # Keycloak, else not offered (the page only takes an API key).
        self.dashboard_url = (config.neuralk_dashboard_url or "").rstrip("/") or None
        dashboard = urlsplit(self.dashboard_url or "")
        self.dashboard_origin = f"{dashboard.scheme}://{dashboard.netloc}" if self.dashboard_url else None
        keycloak = bool(config.neuralk_oidc_client_id and config.neuralk_oidc_client_secret)
        self.sign_in_mode = "dashboard" if self.dashboard_url else ("keycloak" if keycloak else None)
        self.neuralk_sign_in = self.sign_in_mode is not None
        https = self.issuer.startswith("https://")
        # __Host-: only this exact host can set it, so a sibling subdomain
        # cannot plant a known value (it needs Secure, hence https only).
        self.cookie_name = "__Host-seldon_signin" if https else "seldon_signin"
        self._secure_cookie = https
        self._authorization_handler = AuthorizationHandler(self.provider)
        self._oidc_endpoints: dict[str, str] | None = None

    # --- the resource side: what the MCP endpoint needs ---

    @staticmethod
    def issued(credential: str) -> bool:
        """True when a credential is one of this server's blobs (as opposed to a raw API key)."""
        return credential.startswith(TOKEN_PREFIX)

    def api_key_for(self, token: str) -> str | None:
        """The API key an access token carries, or None if it is not a valid access token."""
        data = self.sealer.open("access", token)
        return data["k"] if data else None

    async def key_accepted(self, api_key: str) -> bool:
        """False once Neuralk rejects the key (revoked, expired): the client must sign in again.

        Cached by the auth module, so this costs one call every few minutes per key.
        """
        if self.config.skip_api_key_validation:
            return True
        try:
            await _validate(api_key, self.config)
        except APIKeyAuthError:
            return False
        except _SkipValidation:
            return True
        return True

    def resource_metadata_url(self, path: str) -> str:
        """Where the protected resource metadata of the MCP endpoint at ``path`` lives (RFC 9728)."""
        suffix = "" if path.rstrip("/") == "" else "/mcp"
        return f"{self.issuer}/.well-known/oauth-protected-resource{suffix}"

    def challenge(self, path: str, *, invalid_token: bool) -> str:
        """The WWW-Authenticate value that starts (or restarts) sign-in in an MCP client."""
        params = []
        if invalid_token:
            params += ['error="invalid_token"', 'error_description="The access token is expired or not valid"']
        params += [f'resource_metadata="{self.resource_metadata_url(path)}"', f'scope="{SCOPE}"']
        return "Bearer " + ", ".join(params)

    # --- routes ---

    def routes(self) -> list[Route]:
        token_handler = TokenHandler(self.provider, ClientAuthenticator(self.provider))
        metadata_methods = ["GET", "OPTIONS"]
        return [
            Route(
                "/.well-known/oauth-protected-resource",
                cors_middleware(self._protected_resource_metadata, metadata_methods),
                methods=metadata_methods,
            ),
            Route(
                "/.well-known/oauth-protected-resource/mcp",
                cors_middleware(self._protected_resource_metadata, metadata_methods),
                methods=metadata_methods,
            ),
            Route(
                "/.well-known/oauth-authorization-server",
                cors_middleware(self._authorization_server_metadata, metadata_methods),
                methods=metadata_methods,
            ),
            Route("/authorize", self.authorize, methods=["GET", "POST"]),
            Route("/token", cors_middleware(token_handler.handle, ["POST", "OPTIONS"]), methods=["POST", "OPTIONS"]),
            Route("/register", cors_middleware(self.register, ["POST", "OPTIONS"]), methods=["POST", "OPTIONS"]),
            Route(CONSENT_PATH, self.consent, methods=["GET", "POST"]),
            Route(CALLBACK_PATH, self.callback, methods=["GET"]),
            Route(CONNECT_DESCRIBE_PATH, self.describe, methods=["GET"]),
            Route(CONNECT_COMPLETE_PATH, self.complete, methods=["POST"]),
        ]

    async def _protected_resource_metadata(self, request: Request) -> Response:
        root = request.url.path.rstrip("/") == "/.well-known/oauth-protected-resource"
        return JSONResponse(
            {
                # Must equal the URL the user gives their client: /mcp is the
                # documented one, the root doc describes the bare host.
                "resource": self.issuer if root else self.resource,
                "authorization_servers": [self.issuer],
                "scopes_supported": [SCOPE],
                "bearer_methods_supported": ["header"],
                "resource_name": "Neuralk Seldon",
                "resource_documentation": DOCS_URL,
            }
        )

    async def _authorization_server_metadata(self, request: Request) -> Response:
        return JSONResponse(
            {
                "issuer": self.issuer,
                "authorization_endpoint": f"{self.issuer}/authorize",
                "token_endpoint": f"{self.issuer}/token",
                "registration_endpoint": f"{self.issuer}/register",
                "scopes_supported": [SCOPE],
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
                "token_endpoint_auth_methods_supported": ["none", "client_secret_post"],
                "code_challenge_methods_supported": ["S256"],
                "authorization_response_iss_parameter_supported": True,
                "service_documentation": DOCS_URL,
            }
        )

    async def register(self, request: Request) -> Response:
        """Dynamic client registration (RFC 7591), with nothing stored: the client id is the sealed metadata."""
        try:
            metadata = OAuthClientMetadata.model_validate(await request.json())
        except (ValueError, ValidationError) as exc:
            return _oauth_error("invalid_client_metadata", str(exc).splitlines()[0])
        uris = [str(uri) for uri in metadata.redirect_uris or ()]
        if not uris:
            return _oauth_error("invalid_redirect_uri", "at least one redirect URI is required")
        for uri in uris:
            problem = _redirect_uri_problem(uri)
            if problem:
                return _oauth_error("invalid_redirect_uri", problem)
        # Public clients (Claude, Claude Code) ask for "none". Anything else
        # gets client_secret_post, which RFC 7591 section 3.2.1 lets a server
        # substitute: clients read the method back from this answer. Not
        # client_secret_basic, which the SDK's token handler only accepts with
        # the client id repeated in the form, as few clients send it.
        requested = metadata.token_endpoint_auth_method
        if requested == "private_key_jwt":
            return _oauth_error("invalid_client_metadata", "private_key_jwt is not supported")
        method = "none" if requested == "none" else "client_secret_post"
        if "code" not in metadata.response_types:
            return _oauth_error("invalid_client_metadata", "response_types must include 'code'")

        issued_at = int(time.time())
        name = _clean_name(metadata.client_name)
        client_id = self.sealer.seal("client", {"r": uris, "m": method, "n": name, "t": issued_at}, None)
        body: dict[str, Any] = {
            "client_id": client_id,
            "client_id_issued_at": issued_at,
            "redirect_uris": uris,
            "token_endpoint_auth_method": method,
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "scope": SCOPE,
        }
        if name:
            body["client_name"] = name
        if method != "none":
            body["client_secret"] = self.sealer.client_secret(client_id)
            body["client_secret_expires_at"] = 0
        return JSONResponse(body, status_code=201, headers={"Cache-Control": "no-store"})

    async def authorize(self, request: Request) -> Response:
        """The SDK's authorization endpoint, behind checks it would otherwise answer with a redirect.

        The SDK sends parameter errors back to the client's registered redirect
        URI; registration being open, that would make this server redirect
        anyone anywhere without a click (RFC 9700 section 4.11.2). A request
        that is not a proper PKCE code request gets a page here instead.
        """
        params = request.query_params if request.method == "GET" else await request.form()
        problem = None
        if params.get("response_type") != "code":
            problem = "It asks for something other than an authorization code."
        elif not params.get("code_challenge"):
            problem = "It has no PKCE code challenge."
        elif params.get("code_challenge_method", "S256") != "S256":
            problem = "Its PKCE method is not S256."
        if problem:
            return _message_page("This sign-in link is not valid", f"{problem} Go back to your app and try again.")
        return await self._authorization_handler.handle(request)

    # --- the sign-in page ---

    async def consent(self, request: Request) -> Response:
        """The page the user lands on from their MCP client: sign in, or paste a key."""
        form = await request.form() if request.method == "POST" else None
        pending_token = str((form if form is not None else request.query_params).get("request") or "")
        pending = self.sealer.open("pending", pending_token)
        if pending is None:
            return _expired_page()
        if form is None:
            return self._page(request, pending_token, pending)
        nonce = self._browser_nonce(request)
        if nonce is None or not _same(nonce, str(form.get("csrf") or "")):
            # Not posted from the page this browser was shown: a form submitted
            # from another site, or a browser that drops cookies.
            error = f"Please try again. If this keeps happening, allow cookies for {urlsplit(self.issuer).hostname}."
            return self._page(request, pending_token, pending, error=error, status=400)

        if form.get("action") == "neuralk" and self.sign_in_mode == "dashboard":
            return self._start_dashboard_sign_in(pending_token, nonce)
        if form.get("action") == "neuralk" and self.sign_in_mode == "keycloak":
            try:
                return await self._start_neuralk_sign_in(pending_token, nonce)
            except SignInError as exc:
                return self._page(request, pending_token, pending, error=str(exc), status=502)

        api_key = str(form.get("api_key") or "").strip()
        if not api_key:
            return self._page(request, pending_token, pending, error="Paste your Neuralk API key.", status=400)
        error = await self._check_key(api_key)
        if error:
            return self._page(request, pending_token, pending, error=error, status=400)
        return self._back_to_client(pending, api_key, int(time.time()) + SESSION_TTL_S)

    async def callback(self, request: Request) -> Response:
        """Back from the Neuralk login: turn the user's session into an API key for this connection."""
        params = request.query_params
        upstream = self.sealer.open("upstream", params.get("state"))
        pending = self.sealer.open("pending", upstream["q"]) if upstream else None
        if upstream is None or pending is None:
            return _expired_page()
        nonce = self._browser_nonce(request)
        if nonce is None or not _same(nonce, str(upstream.get("n") or "")):
            return _message_page(
                "This sign-in did not start here",
                "It was started in another browser, or took too long. Go back to your app (Claude, ChatGPT…) "
                "and press Connect again.",
            )
        pending_token = upstream["q"]
        if params.get("error") or not params.get("code"):
            message = "Neuralk sign-in was cancelled. Try again, or use an API key."
            return self._page(request, pending_token, pending, error=message, status=400)
        session_end = int(time.time()) + SESSION_TTL_S
        try:
            user_token = await self._exchange_upstream_code(params["code"], upstream["v"])
            api_key = await self._create_api_key(user_token, pending, session_end)
        except SignInError as exc:
            return self._page(request, pending_token, pending, error=str(exc), status=400)
        return self._back_to_client(pending, api_key, session_end)

    def _browser_nonce(self, request: Request) -> str | None:
        """This browser's sign-in cookie, when it has a well-formed one."""
        value = request.cookies.get(self.cookie_name) or ""
        return value if _NONCE.fullmatch(value) else None

    async def _check_key(self, api_key: str) -> str | None:
        """Why this key cannot be used, or None if it can."""
        if self.config.skip_api_key_validation:
            return None
        try:
            await _validate(api_key, self.config)
        except APIKeyAuthError as exc:
            return str(exc)
        except _SkipValidation:
            return "Neuralk could not be reached to check the key. Try again in a moment."
        return None

    def _back_to_client(self, pending: dict[str, Any], api_key: str, session_end: int) -> Response:
        code = self.provider.issue_code(pending, api_key, session_end)
        return RedirectResponse(
            # iss (RFC 9207): a client talking to several servers can tell which one answered.
            construct_redirect_uri(pending["r"], code=code, state=pending["s"], iss=self.issuer),
            status_code=302,
            headers={"Cache-Control": "no-store"},
        )

    # --- "Continue with Neuralk": the dashboard leg ---

    def _start_dashboard_sign_in(self, pending_token: str, nonce: str) -> Response:
        """Send the user to the dashboard's /connect page, the sign-in sealed in its state."""
        state = self.sealer.seal("upstream", {"q": pending_token, "n": nonce, "m": "dashboard"}, PENDING_TTL_S)
        return RedirectResponse(
            f"{self.dashboard_url}{DASHBOARD_CONNECT_PATH}?{urlencode({'state': state})}", status_code=303
        )

    def _dashboard_flow(self, request: Request, state: str | None) -> tuple[dict[str, Any], dict[str, Any]] | str:
        """The sign-in a dashboard state stands for, or why it cannot go on.

        The browser must be the one this server sent to the dashboard: the
        sign-in cookie comes along on the dashboard's same-site requests, and
        a link built from someone else's sign-in fails here, before any key
        is created.
        """
        upstream = self.sealer.open("upstream", state)
        pending = self.sealer.open("pending", upstream["q"]) if upstream else None
        if upstream is None or pending is None or upstream.get("m") != "dashboard":
            return "This sign-in link has expired. Go back to your app and press Connect again."
        nonce = self._browser_nonce(request)
        if nonce is None or not _same(nonce, str(upstream.get("n") or "")):
            return (
                "This sign-in did not start in this browser, or took too long. Go back to your app "
                "and press Connect again."
            )
        return upstream, pending

    def _dashboard_cors(self, request: Request) -> dict[str, str]:
        """CORS for the dashboard's own origin only, cookies included."""
        origin = request.headers.get("origin")
        if not origin or origin != self.dashboard_origin:
            return {"Vary": "Origin"}
        return {"Access-Control-Allow-Origin": origin, "Access-Control-Allow-Credentials": "true", "Vary": "Origin"}

    async def describe(self, request: Request) -> Response:
        """What the dashboard's /connect page shows and which key it creates."""
        headers = {**self._dashboard_cors(request), "Cache-Control": "no-store"}
        if self.sign_in_mode != "dashboard":
            return JSONResponse({"error": "Dashboard sign-in is not enabled."}, status_code=404, headers=headers)
        flow = self._dashboard_flow(request, request.query_params.get("state"))
        if isinstance(flow, str):
            return JSONResponse({"error": flow}, status_code=400, headers=headers)
        upstream, pending = flow
        redirect_uri = pending["r"]
        session_end = int(time.time()) + SESSION_TTL_S
        return JSONResponse(
            {
                "app": _app_label(redirect_uri),
                "client_name": pending.get("n"),
                "destination": None if urlsplit(redirect_uri).scheme == "https" else redirect_uri,
                "key": {
                    "name": _key_name(pending),
                    "scopes": CREATED_KEY_SCOPES,
                    "expires_at": datetime.fromtimestamp(session_end, tz=timezone.utc).isoformat(),
                },
                # Back to this server's page, e.g. to paste a key instead.
                "back_url": f"{self.issuer}{CONSENT_PATH}?{urlencode({'request': upstream['q']})}",
            },
            headers=headers,
        )

    async def complete(self, request: Request) -> Response:
        """The dashboard posts the key it created (or a cancel): finish the sign-in."""
        if self.sign_in_mode != "dashboard":
            return _message_page("Dashboard sign-in is not enabled", "Go back to your app and try again.")
        form = await request.form()
        flow = self._dashboard_flow(request, str(form.get("state") or ""))
        if isinstance(flow, str):
            return _message_page("This sign-in did not complete", flow)
        upstream, pending = flow
        if form.get("action") == "cancel":
            return RedirectResponse(
                construct_redirect_uri(pending["r"], error="access_denied", state=pending["s"], iss=self.issuer),
                status_code=302,
                headers={"Cache-Control": "no-store"},
            )
        api_key = str(form.get("api_key") or "").strip()
        error = "No API key came back from the dashboard." if not api_key else await self._check_key(api_key)
        if error:
            return self._page(request, upstream["q"], pending, error=error, status=400)
        return self._back_to_client(pending, api_key, int(time.time()) + SESSION_TTL_S)

    # --- "Continue with Neuralk": the Keycloak leg ---

    async def _start_neuralk_sign_in(self, pending_token: str, nonce: str) -> Response:
        endpoints = await self._oidc()
        # PKCE on this leg too: a code stolen from another sign-in cannot be
        # replayed into this one without the verifier sealed in its state. The
        # nonce is the browser cookie the callback checks.
        verifier = secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        state = self.sealer.seal("upstream", {"q": pending_token, "v": verifier, "n": nonce}, PENDING_TTL_S)
        query = urlencode(
            {
                "client_id": self.config.neuralk_oidc_client_id,
                "response_type": "code",
                "redirect_uri": f"{self.issuer}{CALLBACK_PATH}",
                "scope": "openid",
                "state": state,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
        )
        return RedirectResponse(f"{endpoints['authorization_endpoint']}?{query}", status_code=303)

    async def _oidc(self) -> dict[str, str]:
        """The Neuralk identity provider's endpoints, read once from its discovery document."""
        if self._oidc_endpoints is None:
            url = f"{self.config.neuralk_oidc_issuer.rstrip('/')}/.well-known/openid-configuration"
            try:
                async with httpx.AsyncClient(timeout=self.config.api_key_validation_timeout_s) as http:
                    response = await http.get(url)
                    response.raise_for_status()
                    document = response.json()
                self._oidc_endpoints = {
                    "authorization_endpoint": document["authorization_endpoint"],
                    "token_endpoint": document["token_endpoint"],
                }
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                logger.warning("Neuralk sign-in unavailable: discovery at %s failed (%s)", url, exc)
                raise SignInError("Neuralk sign-in is unavailable right now. Use an API key instead.") from exc
        return self._oidc_endpoints

    async def _exchange_upstream_code(self, code: str, verifier: str) -> str:
        endpoints = await self._oidc()
        try:
            async with httpx.AsyncClient(timeout=self.config.api_key_validation_timeout_s) as http:
                response = await http.post(
                    endpoints["token_endpoint"],
                    data={
                        "grant_type": "authorization_code",
                        "code": code,
                        "redirect_uri": f"{self.issuer}{CALLBACK_PATH}",
                        "client_id": self.config.neuralk_oidc_client_id,
                        "client_secret": self.config.neuralk_oidc_client_secret,
                        "code_verifier": verifier,
                    },
                )
        except httpx.HTTPError as exc:
            logger.warning("Neuralk sign-in: token endpoint unreachable (%s)", exc)
            raise SignInError("Neuralk could not be reached to finish signing in. Try again.") from exc
        if response.status_code != 200:
            logger.warning("Neuralk sign-in: token endpoint answered %s", response.status_code)
            raise SignInError("Neuralk sign-in did not complete. Try again, or use an API key.")
        try:
            return str(response.json()["access_token"])
        except (ValueError, KeyError) as exc:
            raise SignInError("Neuralk sign-in did not complete. Try again, or use an API key.") from exc

    async def _create_api_key(self, user_token: str, pending: dict[str, Any], session_end: int) -> str:
        """Create the API key this connection will run on, in the signed-in user's organization.

        It expires with the sign-in, so a connection nobody reopens leaves no
        key behind for long.
        """
        name = _key_name(pending)
        expires_at = datetime.fromtimestamp(session_end, tz=timezone.utc).isoformat()
        url = f"{self.config.neuralk_prediction_url.rstrip('/')}/api/v1/api-keys"
        try:
            async with httpx.AsyncClient(timeout=self.config.api_key_validation_timeout_s) as http:
                response = await http.post(
                    url,
                    json={"name": name, "scopes": CREATED_KEY_SCOPES, "expires_at": expires_at},
                    headers={"Authorization": f"Bearer {user_token}"},
                )
        except httpx.HTTPError as exc:
            logger.warning("Neuralk sign-in: creating the API key failed (%s)", exc)
            raise SignInError("Neuralk could not be reached to create your API key. Try again.") from exc
        if response.status_code == 403:
            detail = _error_message(response)
            if detail and "expired" in detail.lower():
                raise SignInError(detail)
            raise SignInError(
                "Your role in your Neuralk organization cannot create API keys: ask an admin or owner "
                "for a key and paste it below."
            )
        if response.status_code != 201:
            logger.warning("Neuralk sign-in: POST %s answered %s", url, response.status_code)
            raise SignInError("Your API key could not be created. Try again, or use an API key.")
        try:
            return str(response.json()["api_key"])
        except (ValueError, KeyError) as exc:
            raise SignInError("Your API key could not be created. Try again, or use an API key.") from exc

    # --- rendering ---

    def _page(
        self,
        request: Request,
        pending_token: str,
        pending: dict[str, Any],
        *,
        error: str | None = None,
        status: int = 200,
    ) -> Response:
        # One nonce per browser, kept across pages: two sign-ins in two tabs,
        # or a page shown again after Back from Keycloak, all stay valid.
        nonce = self._browser_nonce(request) or secrets.token_urlsafe(24)
        script_nonce = secrets.token_urlsafe(16)
        redirect_uri = pending["r"]
        response = _html(
            _consent_html(
                pending_token=pending_token,
                csrf=nonce,
                app=_app_label(redirect_uri),
                # Where the answer goes, in full, unless it is a web address
                # the app label already names.
                destination=None if urlsplit(redirect_uri).scheme == "https" else redirect_uri,
                client_name=pending.get("n"),
                cancel_url=construct_redirect_uri(
                    redirect_uri, error="access_denied", state=pending["s"], iss=self.issuer
                ),
                sign_in_mode=self.sign_in_mode,
                error=error,
                script_nonce=script_nonce,
            ),
            status,
            script_nonce=script_nonce,
        )
        response.set_cookie(
            self.cookie_name,
            nonce,
            max_age=PENDING_TTL_S,
            path="/",
            secure=self._secure_cookie,
            httponly=True,
            # Lax: sent on the form post from this page and on Keycloak's
            # top-level redirect back, not on a form posted from another site.
            samesite="lax",
        )
        return response


def _key_name(pending: dict[str, Any]) -> str:
    """The name of the key created for a connection, as the dashboard lists it."""
    label = pending.get("n") or urlsplit(pending["r"]).hostname or "MCP client"
    return f"{label[:60]} connector ({datetime.now(timezone.utc):%Y-%m-%d})"


def _error_message(response: httpx.Response) -> str | None:
    """The human message of an API error body, when there is one.

    The platform answers ``{"detail": {"error": {"code", "message", ...}}}``;
    a plain ``{"detail": "..."}`` is read too.
    """
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return None
    if isinstance(detail, dict):
        inner = detail.get("error")
        detail = inner.get("message") if isinstance(inner, dict) else detail.get("message")
    return detail if isinstance(detail, str) else None


def _oauth_error(error: str, description: str, status: int = 400) -> JSONResponse:
    return JSONResponse(
        {"error": error, "error_description": description},
        status_code=status,
        headers={"Cache-Control": "no-store"},
    )


# --- HTML ---


def _security_headers(script_nonce: str | None) -> dict[str, str]:
    scripts = f"script-src 'nonce-{script_nonce}'; " if script_nonce else ""
    return {
        "Cache-Control": "no-store",
        # No framing (clickjacking), nothing loaded from elsewhere, no script
        # but our own. No form-action: browsers apply it to the redirect after
        # the form too, and that redirect goes back to the MCP client.
        "Content-Security-Policy": f"default-src 'none'; {scripts}style-src 'unsafe-inline'; img-src data:; "
        "frame-ancestors 'none'; base-uri 'none'",
        "X-Frame-Options": "DENY",
        # The sealed request is in this page's URL; links out must not carry it.
        "Referrer-Policy": "no-referrer",
    }


# Buttons stay off for a moment after the page shows or regains focus, so a
# click aimed at another window that this one was raised under (double-click
# jacking) lands on nothing. Without JavaScript the buttons simply work.
_ARM_BUTTONS = (
    "(function(){var b=document.querySelectorAll('button'),t;"
    "function arm(){b.forEach(function(x){x.disabled=true});clearTimeout(t);"
    "t=setTimeout(function(){b.forEach(function(x){x.disabled=false})},600)}"
    "arm();window.addEventListener('focus',arm);"
    "document.addEventListener('visibilitychange',function(){if(!document.hidden)arm()})})();"
)

_STYLE = """
:root { --bg:#f6f7f9; --card:#fff; --text:#16181d; --muted:#5b6170; --line:#e3e6eb;
  --accent:#2f4fe0; --accent-text:#fff; --error-bg:#fdecec; --error:#a4231c; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0f1115; --card:#181b21; --text:#eceef2; --muted:#9aa1ae; --line:#2a2e36;
    --accent:#6f86ff; --accent-text:#0f1115; --error-bg:#3a1917; --error:#ffb4ab; } }
* { box-sizing: border-box; }
body { margin:0; min-height:100vh; display:flex; align-items:center; justify-content:center;
  padding:24px 16px; background:var(--bg); color:var(--text);
  font:15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { width:100%; max-width:420px; background:var(--card); border:1px solid var(--line);
  border-radius:14px; padding:28px; }
.brand { font-size:13px; font-weight:600; letter-spacing:.04em; text-transform:uppercase; color:var(--muted); }
h1 { font-size:22px; margin:6px 0 10px; }
p { margin:0 0 14px; color:var(--muted); }
p strong { color:var(--text); }
code { font:13px ui-monospace, SFMono-Regular, Menlo, monospace; color:var(--text); word-break:break-all; }
.error { background:var(--error-bg); color:var(--error); border-radius:8px; padding:10px 12px; margin:0 0 16px; }
button { display:block; width:100%; padding:11px 14px; border-radius:9px; border:1px solid transparent;
  font:inherit; font-weight:600; text-align:center; cursor:pointer; }
button:disabled { opacity:.65; cursor:default; }
.primary { background:var(--accent); color:var(--accent-text); }
.secondary { background:transparent; color:var(--text); border-color:var(--line); }
label { display:block; font-weight:600; margin:0 0 6px; }
input { width:100%; padding:10px 12px; border-radius:9px; border:1px solid var(--line); background:var(--bg);
  color:var(--text); font:14px ui-monospace, SFMono-Regular, Menlo, monospace; margin:0 0 10px; }
.or { display:flex; align-items:center; gap:10px; color:var(--muted); font-size:13px; margin:18px 0; }
.or::before, .or::after { content:""; flex:1; border-top:1px solid var(--line); }
.hint { font-size:13px; margin:8px 0 0; }
a { color:var(--accent); }
.footer { font-size:13px; margin:20px 0 0; }
.cancel { display:inline-block; margin-top:14px; font-size:14px; color:var(--muted); }
"""


def _html(body: str, status: int = 200, *, script_nonce: str | None = None) -> HTMLResponse:
    return HTMLResponse(body, status_code=status, headers=_security_headers(script_nonce))


def _document(title: str, content: str, script_nonce: str | None = None) -> str:
    script = f'<script nonce="{script_nonce}">{_ARM_BUTTONS}</script>' if script_nonce else ""
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head>"
        f"<body><main>{content}</main>{script}</body></html>"
    )


_SIGN_IN_HINTS = {
    "dashboard": "Sign in to your Neuralk dashboard, with your password or a magic link, and approve: "
    "an API key for this connection is created in your organization.",
    # Keycloak's own login page only takes a password: say so, or an account
    # that only ever used magic links gets stuck there.
    "keycloak": "Uses your Neuralk email and password; an API key for this connection is created in your "
    "organization. Only sign in with magic links? Use an API key below.",
}


def _consent_html(
    *,
    pending_token: str,
    csrf: str,
    app: str,
    destination: str | None,
    client_name: str | None,
    cancel_url: str,
    sign_in_mode: str | None,
    error: str | None,
    script_nonce: str,
) -> str:
    e = html.escape
    who = f"<strong>{e(app)}</strong>"
    named = f" (it calls itself “<bdi>{e(client_name)}</bdi>”)" if client_name else ""
    parts = [
        '<div class="brand">Neuralk · Seldon</div>',
        "<h1>Connect Seldon</h1>",
        f"<p>{who}{named} wants to run Seldon predictions with your Neuralk account. "
        "Usage is billed to your organization.</p>",
    ]
    if destination:
        parts.append(f"<p>Access goes to <code>{e(destination)}</code>.</p>")
    if error:
        parts.append(f'<div class="error" role="alert">{e(error)}</div>')
    hidden = (
        f'<input type="hidden" name="request" value="{e(pending_token)}">'
        f'<input type="hidden" name="csrf" value="{e(csrf)}">'
    )
    if sign_in_mode:
        hint = _SIGN_IN_HINTS[sign_in_mode]
        parts.append(
            f'<form method="post" action="{CONSENT_PATH}">{hidden}'
            '<button class="primary" type="submit" name="action" value="neuralk">Continue with Neuralk</button>'
            f'<p class="hint">{hint}</p></form>'
            '<div class="or">or use an API key</div>'
        )
    button_class = "secondary" if sign_in_mode else "primary"
    autofocus = "" if sign_in_mode else " autofocus"
    parts.append(
        f'<form method="post" action="{CONSENT_PATH}">{hidden}'
        '<label for="api_key">Neuralk API key</label>'
        f'<input id="api_key" name="api_key" type="password" placeholder="nk_live_…" autocomplete="off" '
        f'spellcheck="false" required{autofocus}>'
        f'<button class="{button_class}" type="submit" name="action" value="key">Connect with this key</button>'
        f'<p class="hint">No key yet? <a href="{API_KEYS_URL}" target="_blank" rel="noopener noreferrer">'
        "Create one in your dashboard</a>.</p></form>"
    )
    parts.append(
        f'<p class="footer">You will go back to {who} when done. To disconnect later, revoke the key in your '
        f'<a href="{API_KEYS_URL}" target="_blank" rel="noopener noreferrer">dashboard</a>.</p>'
        f'<a class="cancel" href="{e(cancel_url)}">Cancel</a>'
    )
    return _document("Connect Seldon", "".join(parts), script_nonce)


def _message_page(title: str, message: str) -> HTMLResponse:
    return _html(
        _document(
            title,
            f'<div class="brand">Neuralk · Seldon</div><h1>{html.escape(title)}</h1><p>{html.escape(message)}</p>',
        ),
        400,
    )


def _expired_page() -> HTMLResponse:
    return _message_page(
        "This sign-in link has expired", "Go back to your app (Claude, ChatGPT…) and press Connect again."
    )
