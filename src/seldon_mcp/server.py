from __future__ import annotations

import csv
import io
import json
import logging
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
import click
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response

from seldon_mcp.auth import APIKeyAuthError, _SkipValidation, validate_api_key
from seldon_mcp.config import SeldonConfig
from seldon_mcp.downloads import DownloadStore
from seldon_mcp.prediction_client import (
    PredictionAPIError,
    multipart_complete,
    multipart_init,
    multipart_sign,
)

if TYPE_CHECKING:
    from neuralk.exceptions import NeuralkException

# --- Lazy imports ---
#
# ``neuralk`` imports scikit-learn and ``seldon_mcp.dataset`` imports skrub, a
# few seconds of CPU on a small machine. Every tool needs them; startup does
# not, and on a serverless host startup is what the health check times. So
# both are imported the first time a tool touches them. An ``except``
# expression is evaluated only when an exception propagates, so
# ``except _neuralk_exception() as e`` costs nothing until a call fails.


def _dataset():
    from seldon_mcp import dataset

    return dataset


def _sdk():
    from seldon_mcp import neuralk_sdk

    return neuralk_sdk


def _neuralk_exception() -> type[Exception]:
    from neuralk.exceptions import NeuralkException

    return NeuralkException


logger = logging.getLogger("seldon_mcp")


def _sanitize_error(e: Exception, config: SeldonConfig, request_api_key: str | None = None) -> str:
    """Return an error message with any API keys redacted."""
    msg = str(e)
    if config.neuralk_api_key and config.neuralk_api_key in msg:
        msg = msg.replace(config.neuralk_api_key, "***")
    if request_api_key and request_api_key in msg:
        msg = msg.replace(request_api_key, "***")
    return msg


SELDON_MODELS = [
    {"name": "seldon-flash", "description": "Optimized for low latency", "tier": "speed"},
    {"name": "seldon-small", "description": "Balanced speed and accuracy (default)", "tier": "balanced"},
    {"name": "seldon-large", "description": "Maximum accuracy for complex tasks", "tier": "accuracy"},
]

VALID_MODEL_NAMES = {m["name"] for m in SELDON_MODELS}
VALID_TASK_TYPES = {"classification", "regression"}


# Self-describing archive spec returned by create_upload, so a client can build
# the upload WITHOUT reading this server's source.
_UPLOAD_ARCHIVE_SPEC = {
    "container": "a tar archive compressed with zstd (level 6)",
    "files": {
        "metadata.json": {
            "method": "fit_predict",
            "model": "one of seldon-small | seldon-flash | seldon-large",
            "dataset": "<any name>",
            "prompter_config": None,
            "problem_type": "classification | regression",
            "memory_optimization": "true for regression, false for classification",
            "preprocess": True,
            "metadata": {},
            "user": "",
            "version": 1,
        },
        "X_train.npy": "numpy .npy, float32, shape (n_train, n_features)",
        "y_train.npy": "numpy .npy; int64 for classification (label-encode string labels first, "
                       "keep the mapping to decode predictions), float64 for regression",
        "X_test.npy": "numpy .npy, float32, shape (n_test, n_features); the rows to predict on",
    },
    "notes": "Arrays are loaded with allow_pickle=False, so they must be numeric. "
             "Categorical feature columns must be encoded to numbers (e.g. one-hot) before saving. "
             "This layout mirrors the neuralk SDK 1.2.x archive format; if the SDK's format "
             "changes, this spec and the drop_and_predict recipe must be updated to match.",
}


def _validate_params(
    holdout_size: float,
    model: str | None,
    num_rows: int,
    has_separate_file: bool,
    task_type: str | None = None,
) -> None:
    """Validate common tool parameters. Raises ValueError on invalid input."""
    if task_type and task_type not in VALID_TASK_TYPES:
        raise ValueError(
            f"Unknown task_type '{task_type}'. Valid options: {', '.join(sorted(VALID_TASK_TYPES))}"
        )
    if model and model not in VALID_MODEL_NAMES:
        raise ValueError(
            f"Unknown model '{model}'. Valid options: {', '.join(sorted(VALID_MODEL_NAMES))}"
        )
    if not has_separate_file:
        if not 0 < holdout_size < 1:
            raise ValueError(f"holdout_size must be between 0 and 1 (exclusive), got {holdout_size}")
        min_rows = max(2, int(1 / min(holdout_size, 1 - holdout_size)))
        if num_rows < min_rows:
            raise ValueError(
                f"Need at least {min_rows} rows for a {holdout_size} holdout split, got {num_rows}"
            )


_lifespan_config: SeldonConfig | None = None
_download_store: DownloadStore | None = None
_server_key_checked = False


def _init_state(config: SeldonConfig | None = None) -> SeldonConfig:
    """Create the process-wide configuration and download store, once.

    The CLI calls this before serving; the first session's lifespan calls it as
    a fallback (tests and embedders that never go through the CLI). It is
    idempotent: a second call returns what the first one built.

    Nothing built here is ever torn down while the process serves. Over HTTP
    the MCP lifespan runs once per client session — once per request in
    stateless mode — so a teardown at the end of one session would pull the
    download store from under every other session still answering.

    Args:
        config: Configuration to install. Defaults to reading the environment.

    Returns:
        The process-wide configuration.
    """
    global _lifespan_config, _download_store
    if _lifespan_config is None:
        _lifespan_config = config or SeldonConfig()
    if _download_store is None:
        cfg = _lifespan_config
        download_dir = cfg.seldon_download_dir or str(Path(tempfile.gettempdir()) / "seldon-mcp-downloads")
        _download_store = DownloadStore(download_dir, ttl_seconds=cfg.seldon_download_ttl_seconds)
        _download_store.sweep()  # clear anything left over from a previous run
        logger.info("Download store: %s (files live %ss)", download_dir, cfg.seldon_download_ttl_seconds)
    return _lifespan_config


def _start_download_sweeper(store: DownloadStore, interval: int = 60) -> threading.Thread:
    """Start the daemon thread that deletes expired download files.

    A belt to the serve-time braces: files are also deleted when served and
    swept on every save, so the thread only reclaims files nobody fetched.

    Args:
        store: The download store to sweep.
        interval: Seconds between sweeps.

    Returns:
        The started thread.
    """

    def _loop() -> None:
        while True:
            time.sleep(interval)
            try:
                store.sweep()
            except Exception:  # noqa: BLE001 - a sweep must never kill the thread
                logger.exception("download sweep failed")

    thread = threading.Thread(target=_loop, name="seldon-mcp-download-sweeper", daemon=True)
    thread.start()
    return thread


async def _check_server_key_once(config: SeldonConfig) -> None:
    """Pre-validate the server-level key, once per process, log only.

    Per-request keys are validated on the tool call that uses them.
    """
    global _server_key_checked
    if _server_key_checked or not config.neuralk_api_key or config.skip_api_key_validation:
        return
    _server_key_checked = True
    try:
        result = await validate_api_key(
            config.neuralk_api_key,
            base_url=config.neuralk_prediction_url,
            ttl_s=config.api_key_validation_ttl_s,
            timeout_s=config.api_key_validation_timeout_s,
        )
        logger.info(
            "Server-level Neuralk API key validated (org=%s, key=%s, scopes=%s)",
            result.organization_id, result.key_name, ",".join(result.scopes),
        )
    except APIKeyAuthError as exc:
        logger.error("Server-level NEURALK_API_KEY rejected by auth API: %s", exc)
    except _SkipValidation:
        logger.warning(
            "Could not pre-validate server-level NEURALK_API_KEY (auth API unreachable); "
            "validation will be retried on first tool call."
        )


@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Hand each MCP session the process-wide configuration.

    Runs once per session: once per process over stdio, once per client
    session over HTTP, once per request in stateless HTTP mode. State is built
    by ``_init_state`` (idempotent) and deliberately not torn down here.
    """
    config = _init_state()
    await _check_server_key_once(config)
    yield {"config": config}


mcp = FastMCP(
    "seldon",
    instructions="""Seldon is Neuralk's tabular foundation model. It uses in-context learning — \
you provide labeled examples as context and it predicts on new data, with zero hyperparameter tuning.

This server is a thin proxy to Neuralk's prediction API. It never reads your data files from disk — \
it works purely with datasets already uploaded to Neuralk (referenced by a dataset_id) or with data \
passed inline to a tool call. (Predictions too large to return inline are cached briefly on disk and \
served once via a single-use download link.)

Two ways to predict:
1. predict(dataset_key=...) — the recommended path. The dataset has already been uploaded directly to \
Neuralk (returning a dataset_id) by a client/code-execution step or by upload_data. The data never \
passes through this server or the conversation.
2. predict_from_data(data=..., target_column=...) — convenience path for SMALL datasets only. Pass the \
CSV content inline as a string; the server builds the upload archive and predicts in one call. The \
data travels through the tool call, so keep it small (thousands of rows at most). For anything larger, \
upload directly to Neuralk and use predict(dataset_key=...).

IMPORTANT — context selection:
Seldon learns from the context examples you provide, similar to few-shot prompting. The context should \
be RELEVANT to what you're predicting — don't just dump an entire dataset. Irrelevant context adds \
noise and hurts performance.

Key things to know:
- For classification with string labels, predictions are returned in the original label space.
- Three model variants exist: seldon-flash (fast), seldon-small (balanced, default), seldon-large \
(most accurate).
- If auto-detection picks the wrong task type, set task_type="classification" or "regression".""",
    lifespan=app_lifespan,
    dependencies=["polars", "scikit-learn", "skrub", "httpx", "zstandard", "numpy"],
)


API_KEY_HEADER = "x-neuralk-api-key"
# The header MCP hosting platforms (Alpic among them) use for API-key auth.
GENERIC_API_KEY_HEADER = "x-api-key"
AUTHORIZATION_HEADER = "authorization"

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _get_config(ctx: Context) -> SeldonConfig:
    return ctx.request_context.lifespan_context["config"]


def _api_key_from_headers(headers: Any) -> str | None:
    """Read the client's Neuralk API key from HTTP headers.

    Three spellings are accepted: the ``x-neuralk-api-key`` header, the
    generic ``x-api-key`` header that MCP hosting platforms forward, and the
    MCP-conventional ``Authorization: Bearer <key>``, so any client that can
    send one static header can connect.

    Args:
        headers: A mapping with ``get`` (Starlette headers or a plain dict).

    Returns:
        The key, or None when no header carries one.
    """
    for name in (API_KEY_HEADER, GENERIC_API_KEY_HEADER):
        key = headers.get(name)
        if key and key.strip():
            return key.strip()
    auth = headers.get(AUTHORIZATION_HEADER) or headers.get("Authorization") or ""
    scheme, _, token = auth.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    return None


def _get_request_api_key(ctx: Context) -> str | None:
    """Extract the per-request API key from HTTP headers, if present."""
    request = ctx.request_context.request
    if request is not None:
        return _api_key_from_headers(request.headers)
    return None


def _resolve_api_key(ctx: Context, config: SeldonConfig) -> str:
    """Resolve the Neuralk API key: per-request header > server env var.

    With ``require_client_api_key`` set (the hosted deployment), the server's
    own key is never used on a client's behalf: a request without a key is an
    error that names the headers to send.
    """
    header_key = _get_request_api_key(ctx)
    if header_key:
        return header_key

    if config.require_client_api_key:
        raise ValueError(
            "This server needs your own Neuralk API key on every request: send it as the "
            f"'{API_KEY_HEADER}' header or as 'Authorization: Bearer <key>'. Create a key at "
            "https://prediction.neuralk-ai.com/dashboard/api-keys."
        )
    if config.neuralk_api_key:
        return config.neuralk_api_key
    raise ValueError(
        "No Neuralk API key provided. Either set NEURALK_API_KEY on the server "
        f"or pass it via the '{API_KEY_HEADER}' HTTP header (or 'Authorization: Bearer <key>')."
    )

async def _get_api_key(ctx: Context, config: SeldonConfig) -> str:
    """Resolve and validate the Neuralk API key against the saas auth API.

    Validation hits ``GET /api/v1/auth/whoami`` and is cached per-key with a
    short TTL. Network failures fail-open (the SDK call will surface the real
    error). 401/403 raises a ``ValueError`` so the MCP tool returns a clear
    error message to the client instead of a cryptic SDK traceback.
    """
    api_key = _resolve_api_key(ctx, config)

    if config.skip_api_key_validation:
        return api_key

    try:
        await validate_api_key(
            api_key,
            base_url=config.neuralk_prediction_url,
            ttl_s=config.api_key_validation_ttl_s,
            timeout_s=config.api_key_validation_timeout_s,
        )
    except APIKeyAuthError as exc:
        raise ValueError(str(exc)) from exc
    except _SkipValidation:
        # neuralk-saas unreachable; let the SDK surface the real error
        pass

    return api_key


def _sdk_error(e: NeuralkException, config: SeldonConfig, req_key: str | None) -> str:
    """Format a NeuralkException from the SDK into a sanitized JSON error string.

    Terms-of-service rejections get a targeted message pointing at the terms URL,
    since the fix (an org accepting the ToS) is out of this server's hands.
    """
    from neuralk.exceptions import NeuralkTermsNotAcceptedError

    if isinstance(e, NeuralkTermsNotAcceptedError):
        detail = (
            "Neuralk Terms of Service not accepted for this organization. "
            f"Accept version {e.terms_version or '(current)'} at {e.terms_url or 'the Neuralk console'} "
            "before uploading or predicting."
        )
        return json.dumps({"error": detail, "terms_version": e.terms_version, "terms_url": e.terms_url})
    msg = _sanitize_error(e, config, req_key)
    return json.dumps({"error": f"Prediction API error ({getattr(e, 'status_code', '?')}): {msg}"})


def _presigned_error(e: PredictionAPIError, config: SeldonConfig, req_key: str | None) -> str:
    """Format a presigned-flow PredictionAPIError as sanitized JSON.

    The raw multipart flow can't read the SDK's structured terms fields, but a 403
    almost always means the org lacks permission or hasn't accepted the Terms of
    Service — surface that hint so the client isn't left with a bare 403 (parity
    with the friendlier terms message the SDK tools give).
    """
    msg = f"Prediction API error ({e.status_code}): {_sanitize_error(e, config, req_key)}"
    if e.status_code == 403:
        msg += (
            " — this usually means the organization lacks permission or has not accepted "
            "the Neuralk Terms of Service."
        )
    return json.dumps({"error": msg})


async def _build_sdk_client(ctx: Context, config: SeldonConfig):
    """Resolve+validate the API key and construct a neuralk SDK client for this request."""
    api_key = await _get_api_key(ctx, config)
    return _sdk().build_client(api_key, config.neuralk_prediction_url)


# --- Inline-data helpers (no filesystem access) ---


# Predictions flow back through the (size-limited) MCP tool result, so cap how
# many are returned inline. A few thousand numeric predictions fit comfortably;
# callers can raise this, and for very large test sets should predict in chunks.
DEFAULT_MAX_INLINE_PREDICTIONS = 5000


def _truncate_predictions(
    result: dict, predictions: list, probabilities: list | None, max_display: int
) -> None:
    """Attach predictions to a result dict, truncating large arrays inline."""
    result["num_predictions"] = len(predictions)
    result["predictions"] = predictions[:max_display]
    if len(predictions) > max_display:
        result["note"] = (
            f"Showing first {max_display} of {len(predictions)} predictions. Pass a larger "
            "max_predictions to return more; the MCP result is size-limited, so for very large "
            "test sets predict on the data in chunks and concatenate the results client-side."
        )
    if probabilities is not None:
        result["prediction_probabilities"] = probabilities[:max_display]


def _predictions_csv(predictions: list, probabilities: list | None) -> bytes:
    """Serialize the full prediction set to CSV bytes (prediction + prob_* columns)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    n_probs = len(probabilities[0]) if probabilities and probabilities[0] is not None else 0
    writer.writerow(["prediction", *[f"prob_{i}" for i in range(n_probs)]])
    for i, pred in enumerate(predictions):
        row = [pred]
        if probabilities and i < len(probabilities) and probabilities[i] is not None:
            row.extend(probabilities[i])
        writer.writerow(row)
    return buf.getvalue().encode("utf-8")


async def _attach_download(ctx: Context, result: dict, predictions: list, probabilities: list | None) -> None:
    """Write the full predictions to the disk store and add a single-use download_url.

    Only available over HTTP transports (served by the server's /downloads route).
    No-op under stdio, where there is no HTTP endpoint to fetch from.
    """
    request = ctx.request_context.request
    if request is None or not predictions or _download_store is None:
        return
    store = _download_store
    token = await anyio.to_thread.run_sync(lambda: store.save(_predictions_csv(predictions, probabilities)))
    # Behind an ingress the request's own base URL is the pod's view (plain
    # http, whatever Host the proxy forwarded); SELDON_PUBLIC_URL is the address
    # a client can actually fetch from.
    base_url = _get_config(ctx).public_url or str(request.base_url)
    result["download_url"] = f"{base_url.rstrip('/')}/downloads/{token}"
    result["download_note"] = (
        f"All {len(predictions)} predictions as CSV. Single-use link; expires in "
        f"{store.ttl_seconds // 60} min or when downloaded."
    )


def _build_inline_arrays(
    *,
    data: str,
    target_column: str,
    predict_data: str | None,
    feature_columns: list[str] | None,
    model: str | None,
    task_type: str | None,
    holdout_size: float,
    random_state: int,
):
    """Parse inline CSV text into numeric arrays for the upload archive.

    Returns (X_train, y_train, X_test, label_classes, feature_names, problem_type).
    Shared by the inline upload_data and predict_from_data tools. No filesystem access.
    """
    df = _dataset().parse_csv(data)
    _validate_params(
        holdout_size, model, df.height,
        has_separate_file=predict_data is not None, task_type=task_type,
    )
    predict_df = _dataset().parse_csv(predict_data) if predict_data else None
    return _dataset().prepare_arrays(
        df, target_column, predict_df=predict_df, feature_columns=feature_columns,
        holdout_size=holdout_size, random_state=random_state, problem_type=task_type,
    )


# --- MCP Tools ---


# --- Tool annotations ---
#
# The connector directories (Claude, ChatGPT) require every tool to carry a
# title and the read-only / destructive hints; the model uses them to decide
# what needs confirmation. Predictions read the user's data and return an
# answer; uploads create a dataset in the user's own Neuralk account, which is
# neither destructive nor publicly visible.


def _reads(title: str, *, open_world: bool = True) -> ToolAnnotations:
    return ToolAnnotations(
        title=title, readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=open_world
    )


def _writes(title: str) -> ToolAnnotations:
    return ToolAnnotations(
        title=title, readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False
    )


@mcp.tool(annotations=_reads("Predict with Seldon"))
async def predict(
    ctx: Context,
    dataset_key: str,
    label_classes: list[str] | None = None,
    max_predictions: int = DEFAULT_MAX_INLINE_PREDICTIONS,
) -> str:
    """Run inference on a dataset already uploaded to Neuralk's prediction API.

    This is the recommended path: the dataset is uploaded directly to Neuralk
    (returning a dataset_id) by a client/code-execution step or by upload_data;
    this server only references it. No data passes through the server here.

    Args:
        dataset_key: The dataset_id returned by Neuralk's upload API (or upload_data).
        label_classes: Optional ordered class labels used to decode integer-coded
            predictions back to the original labels (returned by upload_data).
        max_predictions: Max predictions to return inline (default 5000). The full
            count is always reported as num_predictions; raise this to return more.
    """
    config = _get_config(ctx)
    req_key = _get_request_api_key(ctx)

    try:
        client = await _build_sdk_client(ctx, config)
        response = await anyio.to_thread.run_sync(
            lambda: _sdk().predict_by_reference(client, dataset_key)
        )

        predictions = _dataset().decode_predictions(response["predictions"], label_classes)
        probabilities = response.get("probabilities")

        result = {
            "source": "prediction_api",
            "dataset_id": dataset_key,
            "request_id": response.get("request_id"),
            "model": response.get("model"),
            "credits_consumed": response.get("credits_consumed"),
            "latency_ms": response.get("latency_ms"),
        }
        await _attach_download(ctx, result, predictions, probabilities)
        _truncate_predictions(result, predictions, probabilities, max_predictions)
        return json.dumps(result, default=str)

    except ValueError as e:
        return json.dumps({"error": _sanitize_error(e, config, req_key)})
    except _neuralk_exception() as e:
        return _sdk_error(e, config, req_key)
    except Exception as e:
        return json.dumps({"error": f"Prediction failed: {_sanitize_error(e, config, req_key)}"})


@mcp.tool(annotations=_reads("Predict from inline data"))
async def predict_from_data(
    ctx: Context,
    data: str,
    target_column: str,
    predict_data: str | None = None,
    feature_columns: list[str] | None = None,
    model: str | None = None,
    task_type: str | None = None,
    holdout_size: float = 0.2,
    random_state: int = 42,
    use_probabilities: bool = True,
    dataset_name: str = "inline",
    max_predictions: int = DEFAULT_MAX_INLINE_PREDICTIONS,
) -> str:
    """Predict from inline CSV data (SMALL datasets only).

    Pass the labeled data as CSV text; the server encodes it into the upload
    archive, uploads it to Neuralk, and returns predictions. The data travels
    through this tool call, so keep it small — for larger data, upload directly
    to Neuralk and use predict(dataset_key=...).

    Args:
        data: Labeled context data as CSV text.
        target_column: Name of the target/label column.
        predict_data: CSV text of unlabeled rows to predict on. If omitted, holds out from data.
        feature_columns: Columns to use as features. If omitted, uses all columns except target.
        model: Seldon model variant (seldon-flash, seldon-small, seldon-large). Defaults to server config.
        task_type: Force "classification" or "regression". If omitted, inferred from the target.
        holdout_size: Fraction to hold out for prediction when predict_data is omitted. Defaults to 0.2.
        random_state: Random seed for the holdout split. Defaults to 42.
        use_probabilities: Include class probabilities in the response when available. Defaults to True.
        dataset_name: Name recorded in the archive metadata. Defaults to "inline".
        max_predictions: Max predictions to return inline (default 5000). num_predictions
            always reports the full count; raise this (or predict in chunks) for larger sets.
    """
    config = _get_config(ctx)
    req_key = _get_request_api_key(ctx)

    try:
        client = await _build_sdk_client(ctx, config)

        X_train, y_train, X_test, label_classes, feature_names, problem_type = _build_inline_arrays(
            data=data, target_column=target_column, predict_data=predict_data,
            feature_columns=feature_columns, model=model, task_type=task_type,
            holdout_size=holdout_size, random_state=random_state,
        )

        resolved_model = model or config.seldon_default_model

        response = await anyio.to_thread.run_sync(
            lambda: _sdk().predict_inline(
                client, X_train=X_train, y_train=y_train, X_test=X_test,
                model=resolved_model, problem_type=problem_type, dataset_name=dataset_name,
            )
        )

        predictions = _dataset().decode_predictions(response["predictions"], label_classes)
        probabilities = response.get("probabilities") if use_probabilities else None

        result = {
            "source": "prediction_api",
            "request_id": response.get("request_id"),
            "model": response.get("model") or resolved_model,
            "task_type": problem_type,
            "num_context_samples": len(X_train),
            "num_predict_samples": len(X_test),
            "feature_columns": feature_names,
            "target_column": target_column,
            "credits_consumed": response.get("credits_consumed"),
            "latency_ms": response.get("latency_ms"),
        }
        await _attach_download(ctx, result, predictions, probabilities)
        _truncate_predictions(result, predictions, probabilities, max_predictions)
        return json.dumps(result, default=str)

    except ValueError as e:
        return json.dumps({"error": _sanitize_error(e, config, req_key)})
    except _neuralk_exception() as e:
        return _sdk_error(e, config, req_key)
    except Exception as e:
        return json.dumps({"error": f"Prediction failed: {_sanitize_error(e, config, req_key)}"})


@mcp.tool(annotations=_writes("Upload a dataset"))
async def upload_data(
    ctx: Context,
    data: str,
    target_column: str,
    predict_data: str | None = None,
    feature_columns: list[str] | None = None,
    model: str | None = None,
    task_type: str | None = None,
    holdout_size: float = 0.2,
    random_state: int = 42,
    dataset_name: str = "inline",
    ttl_days: int | None = None,
) -> str:
    """Upload an inline dataset to Neuralk and return its dataset_id (no inference).

    Use this to upload once and then predict many times via predict(dataset_key=...)
    — e.g. comparing models on the same data. The data is passed inline as CSV text
    (no files), so keep it small; for large datasets upload directly to Neuralk and
    pass the returned dataset_id to predict().

    Returns the dataset_id (pass it to predict() as dataset_key — they are the same
    identifier) plus, for classification targets, the label_classes you should pass
    to predict() to decode integer-coded predictions back to labels.

    Args:
        data: Labeled data as CSV text.
        target_column: Name of the target/label column.
        predict_data: CSV text of unlabeled rows to predict on. If omitted, holds out from data.
        feature_columns: Columns to use as features. If omitted, uses all columns except target.
        model: Seldon model variant recorded in the archive metadata. Defaults to server config.
        task_type: Force "classification" or "regression". If omitted, inferred from the target.
        holdout_size: Fraction to hold out for prediction when predict_data is omitted. Defaults to 0.2.
        random_state: Random seed for the holdout split. Defaults to 42.
        dataset_name: Name recorded in the archive metadata. Defaults to "inline".
        ttl_days: Retention tier in days before Neuralk auto-deletes the dataset —
            one of 1, 7, 30, 90. If omitted, falls back to the server's configured
            default, else Neuralk's own default (90 days). After expiry the
            dataset_id returns a 404.
    """
    config = _get_config(ctx)
    req_key = _get_request_api_key(ctx)

    try:
        client = await _build_sdk_client(ctx, config)
        resolved_ttl = ttl_days if ttl_days is not None else config.seldon_upload_ttl_days
        if resolved_ttl is not None and resolved_ttl not in _sdk().ALLOWED_TTL_DAYS:
            allowed = ", ".join(str(d) for d in _sdk().ALLOWED_TTL_DAYS)
            raise ValueError(f"Invalid ttl_days {resolved_ttl!r}. Allowed retention tiers (days): {allowed}.")

        X_train, y_train, X_test, label_classes, _, problem_type = _build_inline_arrays(
            data=data, target_column=target_column, predict_data=predict_data,
            feature_columns=feature_columns, model=model, task_type=task_type,
            holdout_size=holdout_size, random_state=random_state,
        )

        resolved_model = model or config.seldon_default_model

        upload = await anyio.to_thread.run_sync(
            lambda: _sdk().upload_dataset(
                client, X_train=X_train, y_train=y_train, X_test=X_test,
                model=resolved_model, problem_type=problem_type,
                dataset_name=dataset_name, ttl_days=resolved_ttl,
            )
        )

        result = {
            "source": "prediction_api",
            "dataset_id": upload["dataset_id"],
            "bytes": upload.get("bytes"),
            "etag": upload.get("etag"),
            "ttl_days": upload.get("ttl_days"),
            "model": resolved_model,
            "num_train_samples": len(X_train),
            "num_test_samples": len(X_test),
        }
        if label_classes is not None:
            result["label_classes"] = label_classes
        return json.dumps(result, default=str)

    except ValueError as e:
        return json.dumps({"error": _sanitize_error(e, config, req_key)})
    except _neuralk_exception() as e:
        return _sdk_error(e, config, req_key)
    except Exception as e:
        return json.dumps({"error": f"Upload failed: {_sanitize_error(e, config, req_key)}"})


@mcp.tool(annotations=_writes("Start a multipart upload"))
async def create_upload(ctx: Context, num_parts: int = 1) -> str:
    """Begin a presigned upload to Neuralk and return URL(s) to upload bytes to.

    Use this to upload a real FILE without the data passing through this server or
    the conversation. The Neuralk API key stays on the server — the returned
    presigned URLs require NO key. Flow: call this, build the tar+zstd archive and
    PUT it to the returned url(s) (capturing each ETag), then call complete_upload,
    then predict(dataset_key). See the `drop_and_predict` prompt for the full recipe.

    Args:
        num_parts: Number of upload parts. Use 1 for archives under 5GB (a single
            PUT); use more only for very large multipart uploads.

    The response is self-describing: it includes `archive_spec` (exactly what to
    build) and `recipe` (ready-to-run Python with the presigned URL filled in), so
    you do not need this server's source code to perform the upload.

    Returns {dataset_key, upload_id, parts, expires_seconds, archive_spec, recipe, next_steps}.
    """
    config = _get_config(ctx)
    req_key = _get_request_api_key(ctx)
    try:
        api_key = await _get_api_key(ctx, config)
        key = f"auto/{uuid.uuid4().hex}.tar.zst"
        init = await multipart_init(base_url=config.neuralk_prediction_url, api_key=api_key, key=key)
        signed = await multipart_sign(
            base_url=config.neuralk_prediction_url, api_key=api_key,
            upload_id=init["upload_id"], key=key, part_count=num_parts,
        )
        parts = signed.get("parts") or []
        first_url = parts[0]["url"] if parts else "<presigned url>"
        recipe = (
            "pip install numpy pandas zstandard\n\n"
            + _DROP_AND_PREDICT_SNIPPET
            .replace("__FILE__", "<path to your CSV>")
            .replace("__TARGET__", "<target column>")
            .replace("<paste parts[0].url from create_upload>", first_url)
        )
        return json.dumps({
            "dataset_key": key,
            "upload_id": init["upload_id"],
            "parts": parts,
            "expires_seconds": signed.get("expires_seconds"),
            "archive_spec": _UPLOAD_ARCHIVE_SPEC,
            "recipe": recipe,
            "next_steps": (
                "1) Build the tar+zstd archive per archive_spec (or run `recipe`), setting "
                "PROBLEM_TYPE and MODEL. 2) PUT the archive bytes to parts[0].url and capture the "
                "ETag response header. 3) call complete_upload(upload_id, dataset_key, "
                'parts=[{"part_number": 1, "etag": <etag>}]). 4) call predict(dataset_key, '
                "label_classes) — pass label_classes from the recipe output to decode predictions."
            ),
        })
    except ValueError as e:
        return json.dumps({"error": _sanitize_error(e, config, req_key)})
    except PredictionAPIError as e:
        return _presigned_error(e, config, req_key)
    except Exception as e:
        return json.dumps({"error": f"Create upload failed: {_sanitize_error(e, config, req_key)}"})


@mcp.tool(annotations=_writes("Complete a multipart upload"))
async def complete_upload(ctx: Context, upload_id: str, dataset_key: str, parts: list[dict]) -> str:
    """Finalize a presigned upload started with create_upload.

    Args:
        upload_id: The upload_id returned by create_upload.
        dataset_key: The dataset_key returned by create_upload.
        parts: Ordered list of {"part_number": int, "etag": str} from your PUT
            responses (the ETag response header of each part upload).

    Returns {dataset_key} to pass to predict().
    """
    config = _get_config(ctx)
    req_key = _get_request_api_key(ctx)
    try:
        api_key = await _get_api_key(ctx, config)
        result = await multipart_complete(
            base_url=config.neuralk_prediction_url, api_key=api_key,
            upload_id=upload_id, key=dataset_key, parts=parts,
        )
        return json.dumps({"dataset_key": result.get("key", dataset_key), "location": result.get("location")})
    except ValueError as e:
        return json.dumps({"error": _sanitize_error(e, config, req_key)})
    except PredictionAPIError as e:
        return _presigned_error(e, config, req_key)
    except Exception as e:
        return json.dumps({"error": f"Complete upload failed: {_sanitize_error(e, config, req_key)}"})


@mcp.tool(annotations=_reads("List Seldon models", open_world=False))
async def list_models() -> str:
    """List available Seldon model variants and their characteristics.

    Returns the three Seldon model options: seldon-flash (speed),
    seldon-small (balanced), and seldon-large (accuracy).
    """
    return json.dumps(SELDON_MODELS)


# --- MCP Resources ---


@mcp.resource("seldon://models")
async def get_available_models() -> str:
    """Available Seldon model variants and their descriptions."""
    return json.dumps(SELDON_MODELS)


@mcp.resource("seldon://config")
async def get_config_resource() -> str:
    """Current Seldon MCP server configuration (API key masked)."""
    config = _lifespan_config
    if config is None:
        return json.dumps({"error": "Server not fully initialized"})
    return json.dumps({
        "default_model": config.seldon_default_model,
        "prediction_url": config.neuralk_prediction_url,
        "api_key": "set" if config.neuralk_api_key else "not set",
    })


# --- MCP Prompts ---


_DROP_AND_PREDICT_SNIPPET = '''import io, json, tarfile, urllib.request
import numpy as np, pandas as pd, zstandard as zstd

CSV_PATH = "__FILE__"
TARGET = "__TARGET__"
PROBLEM_TYPE = "classification"   # set to "regression" for continuous targets
MODEL = "seldon-small"            # seldon-small | seldon-flash | seldon-large
PRESIGNED_URL = "<paste parts[0].url from create_upload>"
HOLDOUT = 0.2

df = pd.read_csv(CSV_PATH)
y_raw = df[TARGET].to_numpy()
X = pd.get_dummies(df.drop(columns=[TARGET])).to_numpy(dtype=np.float32)

# y dtype must match the task: float64 for regression, int64 for classification
if PROBLEM_TYPE == "regression":
    y, label_classes = y_raw.astype(np.float64), None
elif y_raw.dtype.kind in "OUS":           # string labels -> integer codes
    classes, codes = np.unique(y_raw, return_inverse=True)
    y, label_classes = codes.astype(np.int64), classes.tolist()
else:
    y, label_classes = y_raw.astype(np.int64), None

# hold out a RANDOM subset to predict on; train on the rest. Shuffling avoids the
# order bias of taking the first rows. Note: feature encoding here (pd.get_dummies)
# is simpler than the server's predict_from_data path (skrub TableVectorizer), so
# results may differ slightly between the two entry points.
rng = np.random.default_rng(0)
perm = rng.permutation(len(X))
k = max(1, int(len(X) * HOLDOUT))
test_idx, train_idx = perm[:k], perm[k:]
X_test, X_train, y_train = X[test_idx], X[train_idx], y[train_idx]

def npy(a):
    b = io.BytesIO(); np.save(b, np.ascontiguousarray(a), allow_pickle=False); return b.getvalue()

meta = {"method": "fit_predict", "model": MODEL, "dataset": "dropped", "prompter_config": None,
        "problem_type": PROBLEM_TYPE, "memory_optimization": PROBLEM_TYPE == "regression",
        "preprocess": True, "metadata": {}, "user": "", "version": 1}
members = {"metadata.json": json.dumps(meta).encode(),
          "X_train.npy": npy(X_train), "y_train.npy": npy(y_train), "X_test.npy": npy(X_test)}

raw = io.BytesIO()
with tarfile.open(fileobj=raw, mode="w") as tar:
    for name, data in members.items():
        ti = tarfile.TarInfo(name); ti.size = len(data); tar.addfile(ti, io.BytesIO(data))
archive = zstd.ZstdCompressor(level=6).compress(raw.getvalue())

req = urllib.request.Request(PRESIGNED_URL, data=archive, method="PUT")
with urllib.request.urlopen(req) as resp:
    etag = resp.headers.get("ETag")
print(json.dumps({"etag": etag, "label_classes": label_classes}))
'''


@mcp.prompt()
def drop_and_predict(file_path: str, target_column: str) -> str:
    """Recipe to upload a dropped data file to Neuralk and predict (file-free server)."""
    snippet = _DROP_AND_PREDICT_SNIPPET.replace("__FILE__", file_path).replace("__TARGET__", target_column)
    return (
        "This MCP server never reads files. To predict on the dropped file, the upload is done "
        "by YOU in code execution (the file never reaches this server), and the Neuralk API key "
        "stays on the server. Steps:\n\n"
        "1. Call the `create_upload` tool. It returns `dataset_key`, `upload_id`, and "
        "`parts` (a list with a presigned upload URL at `parts[0].url`).\n\n"
        "2. In code execution, install deps and run the script below — paste `parts[0].url` into "
        "PRESIGNED_URL, and set PROBLEM_TYPE (\"classification\" or \"regression\") and MODEL "
        "(seldon-small/seldon-flash/seldon-large). It builds the tar+zstd archive from the file and uploads "
        "it directly to storage (no key needed), printing the `etag` and `label_classes`:\n"
        "```bash\npip install numpy pandas zstandard\n```\n"
        "```python\n" + snippet + "```\n\n"
        "3. Call `complete_upload` with `upload_id`, `dataset_key`, and "
        '`parts=[{"part_number": 1, "etag": <etag from step 2>}]`.\n\n'
        "4. Call `predict` with `dataset_key` and the `label_classes` from step 2 to get "
        "predictions (decoded back to the original labels).\n\n"
        "For very large files, request more parts from `create_upload` and PUT each part "
        "(min 5MB each) to its URL. For a small dataset you can skip all this and call "
        "`predict_from_data` with the CSV text inline instead."
    )


# --- HTTP routes ---


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(request: Request) -> Response:
    """Liveness and readiness probe: the process is up and serving."""
    return PlainTextResponse("ok")


@mcp.custom_route("/downloads/{token}", methods=["GET"])
async def download_predictions(request: Request) -> Response:
    """Serve a full prediction CSV once, then delete it (single-use, expiring link)."""
    store = _download_store
    if store is None:
        return PlainTextResponse("Downloads not available.", status_code=404)
    token = request.path_params["token"]
    data = await anyio.to_thread.run_sync(store.take, token)
    if data is None:
        return PlainTextResponse("Not found, already downloaded, or expired.", status_code=404)
    return Response(
        content=data,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="predictions.csv"'},
    )


class RequireClientKeyMiddleware:
    """Refuse tool calls that carry no client API key (ASGI middleware).

    The hosted deployment sets ``REQUIRE_CLIENT_API_KEY=true``: this server
    holds no Neuralk key of its own, so a ``tools/call`` without one cannot do
    anything useful and is answered ``401`` up front, naming the accepted
    headers. Discovery stays open: ``initialize``, ``tools/list`` and the other
    read-only protocol methods answer without a key, so directories, catalogs
    and audit tools can read the tool list before a user has configured a
    key. ``/healthz`` and the single-use ``/downloads`` links are not MCP
    requests and pass through.
    """

    GATED_METHODS = frozenset({"tools/call"})

    def __init__(self, app: Any, mcp_paths: set[str]) -> None:
        self.app = app
        self.mcp_paths = {p.rstrip("/") or "/" for p in mcp_paths}

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "").rstrip("/") or "/"
        if path not in self.mcp_paths or _api_key_from_headers(Headers(scope=scope)):
            await self.app(scope, receive, send)
            return

        # No key: read the JSON-RPC body to see whether it is a tool call, then
        # hand the app a receive() that replays what was consumed.
        chunks: list[bytes] = []
        more = True
        while more:
            message = await receive()
            if message["type"] == "http.request":
                chunks.append(message.get("body", b""))
                more = message.get("more_body", False)
            else:  # http.disconnect
                more = False
        body = b"".join(chunks)

        if self._is_gated(body):
            response = JSONResponse(
                {
                    "error": (
                        "A Neuralk API key is required to call tools. Send it as the "
                        f"'{API_KEY_HEADER}' header or as 'Authorization: Bearer <key>'. "
                        "Create a key at https://prediction.neuralk-ai.com/dashboard/api-keys."
                    )
                },
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="neuralk"'},
            )
            await response(scope, receive, send)
            return

        replayed = False

        async def replay() -> dict[str, Any]:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)

    @classmethod
    def _is_gated(cls, body: bytes) -> bool:
        """True when the JSON-RPC body (single message or batch) contains a gated method."""
        try:
            payload = json.loads(body)
        except ValueError:
            return False  # let the transport produce the parse error
        messages = payload if isinstance(payload, list) else [payload]
        return any(isinstance(m, dict) and m.get("method") in cls.GATED_METHODS for m in messages)


def _configure_binding(host: str, port: int) -> None:
    """Point FastMCP at the bind address and fix its transport security for it.

    FastMCP turns DNS-rebinding protection on at construction because its
    default host is loopback, and that protection accepts only loopback Host
    headers. Behind an ingress every request carries the public name, so on a
    non-loopback bind the check is turned off: the ingress in front is what
    decides who reaches this process. On loopback the protection stays.
    """
    mcp.settings.host = host
    mcp.settings.port = port
    if host not in _LOOPBACK_HOSTS:
        mcp.settings.transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)


def build_http_app(
    host: str = "0.0.0.0",
    port: int = 8000,
    *,
    stateless: bool = True,
    json_response: bool = True,
):
    """Build the ASGI app of the hosted server.

    Serves the Streamable HTTP MCP endpoint at ``/mcp`` and at ``/`` (so a
    client given either the bare host or the ``/mcp`` URL works), ``/healthz``
    for probes and ``/downloads/<token>`` for large prediction sets. With
    ``require_client_api_key`` set, MCP requests without a key are refused.

    Args:
        host: Address the server will bind (decides the transport security).
        port: Port the server will bind.
        stateless: No session state between requests, so any replica can
            answer any request — the mode to run behind a load balancer.
        json_response: Answer tool calls with a JSON body instead of an event
            stream; simpler through proxies that buffer.

    Returns:
        The Starlette application.
    """
    from starlette.routing import Route

    config = _init_state()
    _configure_binding(host, port)
    mcp.settings.stateless_http = stateless
    mcp.settings.json_response = json_response

    app = mcp.streamable_http_app()
    mcp_path = mcp.settings.streamable_http_path
    if mcp_path != "/":
        mcp_route = next(r for r in app.routes if getattr(r, "path", None) == mcp_path)
        # Reuse the exact same ASGI endpoint so / behaves identically to /mcp.
        app.router.routes.append(Route("/", endpoint=mcp_route.endpoint))
    if config.require_client_api_key:
        app.add_middleware(RequireClientKeyMiddleware, mcp_paths={mcp_path, "/"})
    return app


def _warm_imports() -> None:
    """Import the heavy dependencies the tools need (see the lazy imports above).

    Run in a background thread once the server is up: the health check and
    the first ``initialize`` do not wait for scikit-learn and skrub, and the
    first tool call finds them loaded instead of paying for the import inside
    its own time budget (a serverless host cuts a tool call after 30 s, and a
    cold import can take most of that on a small machine).
    """
    try:
        t0 = time.monotonic()
        _sdk()
        t1 = time.monotonic()
        _dataset()
        t2 = time.monotonic()
        logger.info("Tool imports warmed: neuralk %.1fs, dataset %.1fs", t1 - t0, t2 - t1)
    except Exception:  # pragma: no cover - a broken import surfaces on the first tool call anyway
        logger.exception("Warming the tool imports failed")


def _start_import_warmer() -> threading.Thread:
    thread = threading.Thread(target=_warm_imports, name="seldon-import-warmer", daemon=True)
    thread.start()
    return thread


def _serve_streamable_http(
    host: str, port: int, *, stateless: bool, json_response: bool, forwarded_allow_ips: str
) -> None:
    """Run the hosted server under uvicorn."""
    import uvicorn

    app = build_http_app(host, port, stateless=stateless, json_response=json_response)
    assert _download_store is not None  # built by build_http_app
    _start_download_sweeper(_download_store)
    _start_import_warmer()
    logger.info(
        "Serving Streamable HTTP on %s:%s (stateless=%s, json_response=%s, client key required=%s)",
        host, port, stateless, json_response, bool(_lifespan_config and _lifespan_config.require_client_api_key),
    )
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=mcp.settings.log_level.lower(),
        proxy_headers=True,
        forwarded_allow_ips=forwarded_allow_ips,
    )


def serve(
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    stateless: bool = True,
    json_response: bool = True,
    forwarded_allow_ips: str = "127.0.0.1",
) -> None:
    """Start the server on the given transport; the entry point behind the CLI.

    Args:
        transport: ``stdio``, ``sse`` or ``streamable-http``.
        host: Bind address for the HTTP transports.
        port: Bind port for the HTTP transports.
        stateless: Streamable HTTP only, see :func:`build_http_app`.
        json_response: Streamable HTTP only, see :func:`build_http_app`.
        forwarded_allow_ips: Proxies whose ``X-Forwarded-*`` headers are trusted.
    """
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    _init_state()

    if transport == "streamable-http":
        _serve_streamable_http(
            host, port, stateless=stateless, json_response=json_response, forwarded_allow_ips=forwarded_allow_ips
        )
        return
    if transport != "stdio":
        _configure_binding(host, port)
    mcp.run(transport=transport)


# --- CLI ---


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse", "streamable-http"]),
    default="stdio",
    help="MCP transport to use.",
)
@click.option("--host", type=str, default="127.0.0.1", help="Host to bind for SSE/HTTP transport.")
@click.option("--port", type=int, default=8000, help="Port for SSE/HTTP transport.")
@click.option(
    "--stateless/--stateful",
    default=True,
    help="Streamable HTTP only: keep no session state between requests, so any replica can answer "
    "any request (the mode to run behind a load balancer). Default: stateless.",
)
@click.option(
    "--json-response/--sse-response",
    default=True,
    help="Streamable HTTP only: answer tool calls with a plain JSON body instead of an event stream. "
    "Default: JSON.",
)
@click.option(
    "--forwarded-allow-ips",
    type=str,
    default="127.0.0.1",
    help="Proxies whose X-Forwarded-* headers are trusted (uvicorn's setting). "
    "Use '*' behind an ingress you control.",
)
def main(
    transport: str,
    host: str,
    port: int,
    stateless: bool,
    json_response: bool,
    forwarded_allow_ips: str,
) -> None:
    """Start the Seldon MCP server."""
    serve(
        transport,
        host,
        port,
        stateless=stateless,
        json_response=json_response,
        forwarded_allow_ips=forwarded_allow_ips,
    )


if __name__ == "__main__":
    main()
