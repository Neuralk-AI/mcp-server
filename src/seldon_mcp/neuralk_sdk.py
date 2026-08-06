"""Thin wrapper over the official ``neuralk`` SDK (>=1.2.0) for the inline and
upload-by-reference prediction paths.

Since 1.2.0 the SDK exposes the upload/inference-by-reference workflow we used
to hand-roll over raw HTTP (``NeuralkAPI.datasets.create`` returns a reusable
``dataset_id``; ``classifications``/``regressions.create`` run inference either
inline or by ``dataset_key``). We delegate to it here so the server no longer
tracks the wire format itself.

The SDK is synchronous (it uses ``httpx.Client`` internally), so the async MCP
tools call these helpers via ``anyio.to_thread.run_sync``. The functions here do
no async work and hold no state beyond the client they are handed.

Note: the presigned multipart upload flow (create_upload/complete_upload) has no
SDK equivalent and still lives in ``prediction_client.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from neuralk import NeuralkAPI

# The SDK treats ANY explicitly-supplied host as on-premise, and refuses dataset
# uploads on such hosts. So we only forward a host when the configured URL is
# genuinely custom; the default cloud endpoint is left implicit (SDK default),
# which keeps uploads working.
DEFAULT_CLOUD_URL = "https://api.prediction.neuralk-ai.com"

# Retention tiers the API accepts, mirrored from neuralk._api.ALLOWED_TTL_DAYS so
# we can validate/report before calling the SDK (which raises ValueError itself).
ALLOWED_TTL_DAYS = (1, 7, 30, 90)


def build_client(api_key: str, prediction_url: str) -> NeuralkAPI:
    """Construct a NeuralkAPI client for the configured endpoint.

    Passes ``host`` only for a non-default (on-premise) URL; the default cloud
    endpoint is left implicit so the SDK's cloud-only upload path stays enabled.
    """
    host = None if prediction_url.rstrip("/") == DEFAULT_CLOUD_URL else prediction_url
    return NeuralkAPI(api_key=api_key, host=host)


def _to_list(value: Any) -> Any:
    """Normalize numpy arrays (and array-likes) to JSON-serializable lists."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def _normalize(response: dict[str, Any]) -> dict[str, Any]:
    """Turn an SDK response into plain JSON-serializable values.

    The SDK may return predictions/probabilities as numpy arrays; convert them to
    lists so the server can json.dumps the result without stringifying arrays.
    """
    return {
        "predictions": _to_list(response.get("predictions")) or [],
        "probabilities": _to_list(response.get("probabilities")),
        "request_id": response.get("request_id"),
        "model": response.get("model"),
        "credits_consumed": response.get("credits_consumed"),
        "latency_ms": response.get("latency_ms"),
    }


def upload_dataset(
    client: NeuralkAPI,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    model: str,
    problem_type: str,
    dataset_name: str,
    ttl_days: int | None = None,
) -> dict[str, Any]:
    """Upload a dataset and return its raw SDK response ({dataset_id, bytes, etag, ttl_days})."""
    return client.datasets.create(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        model=model,
        problem_type=problem_type,
        dataset_name=dataset_name,
        ttl_days=ttl_days,
    )


def predict_inline(
    client: NeuralkAPI,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    model: str,
    problem_type: str,
    dataset_name: str,
) -> dict[str, Any]:
    """Run inference on inline arrays in a single call, routed by problem type."""
    resource = client.classifications if problem_type == "classification" else client.regressions
    response = resource.create(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        model=model,
        problem_type=problem_type,
        dataset_name=dataset_name,
    )
    return _normalize(response)


def predict_by_reference(client: NeuralkAPI, dataset_key: str) -> dict[str, Any]:
    """Run inference on an already-uploaded dataset by its dataset_key.

    The stored archive already carries the task type and settings, so the request
    body is empty and the resource namespace is irrelevant — the server routes on
    the archive metadata. (Passing any request setting alongside dataset_key would
    be rejected by the SDK, so we pass only the key.)
    """
    response = client.classifications.create(dataset_key=dataset_key)
    return _normalize(response)
