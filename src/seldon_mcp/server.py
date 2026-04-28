from __future__ import annotations

import json
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any

import anyio
import click
from mcp.server.fastmcp import Context, FastMCP
from neuralk import NeuralkException, Seldon, SeldonClassifier, SeldonRegressor
from sklearn.model_selection import train_test_split

from seldon_mcp.auth import APIKeyAuthError, _SkipValidation, validate_api_key
from seldon_mcp.config import SeldonConfig
from seldon_mcp.data_loader import describe_dataframe, load_dataframe
from seldon_mcp.metrics import compute_classification_metrics, compute_regression_metrics

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


@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Initialize configuration at server startup."""
    global _lifespan_config
    config = SeldonConfig()
    _lifespan_config = config
    logger.info("Seldon MCP server started")
    if config.neuralk_api_key and not config.skip_api_key_validation:
        try:
            result = await validate_api_key(
                config.neuralk_api_key,
                base_url=config.neuralk_api_base_url,
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
    try:
        yield {"config": config}
    finally:
        _lifespan_config = None
        logger.info("Seldon MCP server shutting down")


mcp = FastMCP(
    "seldon",
    instructions="""Seldon is Neuralk's tabular foundation model. It uses in-context learning — \
you provide labeled examples as context and it predicts on new data, with zero hyperparameter tuning.

IMPORTANT — context selection:
Seldon is NOT a traditional model where more data is always better. It learns from the context \
examples you provide, similar to few-shot prompting. The context should be RELEVANT to what you're \
predicting. For example, if predicting churn rate for a specific customer category, provide context \
examples from that category or very similar ones — don't just dump the entire dataset. \
Irrelevant context examples add noise and hurt performance. When helping users, think about what \
subset of their data is most representative of the prediction target and guide them to filter or \
select context accordingly.

Typical workflow:
1. Use describe_data to inspect the dataset (shape, columns, types, nulls).
2. Think about which rows are most relevant as context for the prediction task. Help the user \
filter or select appropriate context data if needed.
3. Use predict or evaluate with the context file and target column.
4. If auto-detection picks the wrong task type, set task_type="classification" or "regression" explicitly.

Key things to know:
- context_file is the labeled data Seldon learns from. Quality and relevance of context matters \
more than quantity.
- If you omit predict_file/test_file, a holdout split is used automatically.
- File paths are relative to the server's data directory. Ask the user for the file path if not provided.
- Three model variants exist: seldon-flash (fast), seldon-small (balanced, default), seldon-large (most accurate).
- For classification, evaluate returns accuracy, F1, precision, recall, and a confusion matrix.
- For regression, evaluate returns MAE, RMSE, R2, and median absolute error.
- Always describe the data first so you can identify the correct target column and understand the features.""",
    lifespan=app_lifespan,
    dependencies=["neuralk", "polars", "scikit-learn"],
)


API_KEY_HEADER = "x-neuralk-api-key"


def _get_config(ctx: Context) -> SeldonConfig:
    return ctx.request_context.lifespan_context["config"]


def _get_request_api_key(ctx: Context) -> str | None:
    """Extract the per-request API key from HTTP headers, if present."""
    request = ctx.request_context.request
    if request is not None:
        return request.headers.get(API_KEY_HEADER)
    return None


def _resolve_api_key(ctx: Context, config: SeldonConfig) -> str:
    """Resolve the Neuralk API key: per-request header > server env var."""
    header_key = _get_request_api_key(ctx)
    if header_key:
        return header_key

    if config.neuralk_api_key:
        return config.neuralk_api_key

    raise ValueError(
        "No Neuralk API key provided. Either set NEURALK_API_KEY on the server "
        f"or pass it via the '{API_KEY_HEADER}' HTTP header."
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
            base_url=config.neuralk_api_base_url,
            ttl_s=config.api_key_validation_ttl_s,
            timeout_s=config.api_key_validation_timeout_s,
        )
    except APIKeyAuthError as exc:
        raise ValueError(str(exc)) from exc
    except _SkipValidation:
        # neuralk-saas unreachable; let the SDK surface the real error
        pass

    return api_key


VALID_TASK_TYPES = {"classification", "regression"}


async def _make_seldon(
    ctx: Context, config: SeldonConfig, model: str | None = None, task_type: str | None = None,
) -> Seldon | SeldonClassifier | SeldonRegressor:
    """Create a Seldon instance with the resolved API key.

    If task_type is specified, returns the corresponding estimator directly.
    Otherwise returns the auto-detecting Seldon convenience class.
    """
    api_key = await _get_api_key(ctx, config)
    kwargs: dict[str, Any] = {"api_key": api_key}
    if config.neuralk_host:
        kwargs["host"] = config.neuralk_host
    if model:
        kwargs["model"] = model
    else:
        kwargs["model"] = config.seldon_default_model

    if task_type == "classification":
        return SeldonClassifier(**kwargs)
    elif task_type == "regression":
        return SeldonRegressor(**kwargs)
    return Seldon(**kwargs)


def _get_task_type(seldon: Seldon) -> str:
    """Get the task type from a fitted Seldon model."""
    task = getattr(seldon, "task_type_", None)
    if task:
        return str(task)
    # Fallback: check which estimator was dispatched
    if hasattr(seldon, "classes_"):
        return "classification"
    return "regression"


def _prepare_data(
    df,
    target_column: str,
    feature_columns: list[str] | None = None,
):
    """Extract features and target from a polars DataFrame, returning pandas objects."""
    pdf = df.to_pandas()
    if target_column not in pdf.columns:
        available = ", ".join(pdf.columns.tolist())
        raise ValueError(f"Target column '{target_column}' not found. Available columns: {available}")

    if feature_columns:
        missing = [c for c in feature_columns if c not in pdf.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {', '.join(missing)}")
        X = pdf[feature_columns]
    else:
        X = pdf.drop(columns=[target_column])

    y = pdf[target_column].values
    return X, y


# --- MCP Tools ---


@mcp.tool()
async def describe_data(ctx: Context, file_path: str, include_sample: bool = True) -> str:
    """Load a tabular data file and return a statistical summary.

    Describes the shape, column types, null counts, numeric statistics,
    and optionally sample rows from a CSV, Excel, Parquet, or JSON file.

    Args:
        file_path: Path to the data file (CSV, Excel, Parquet, or JSON).
        include_sample: Whether to include the first 5 rows as a sample. Defaults to True.
    """
    config = _get_config(ctx)
    try:
        df = await anyio.to_thread.run_sync(
            lambda: load_dataframe(file_path, config.seldon_data_dir)
        )
        summary = describe_dataframe(df, include_sample=include_sample)
        summary["file_path"] = file_path
        return json.dumps(summary, default=str)
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def predict(
    ctx: Context,
    context_file: str,
    target_column: str,
    predict_file: str | None = None,
    feature_columns: list[str] | None = None,
    model: str | None = None,
    task_type: str | None = None,
    holdout_size: float = 0.2,
    random_state: int = 42,
) -> str:
    """Make predictions using Seldon's tabular foundation model.

    Provide a context file with labeled examples and optionally a separate file
    to predict on. If no predict_file is given, a holdout split from the context
    file is used.

    Args:
        context_file: Path to the labeled data file used as context examples.
        target_column: Name of the target/label column.
        predict_file: Path to unlabeled data to predict on. If omitted, holds out from context_file.
        feature_columns: Columns to use as features. If omitted, uses all columns except target.
        model: Seldon model variant (seldon-flash, seldon-small, seldon-large). Defaults to server config.
        task_type: Force "classification" or "regression". If omitted, Seldon auto-detects from the target.
        holdout_size: Fraction to hold out for prediction when predict_file is omitted. Defaults to 0.2.
        random_state: Random seed for the holdout split. Defaults to 42.
    """
    config = _get_config(ctx)
    req_key = _get_request_api_key(ctx)

    try:
        context_df = await anyio.to_thread.run_sync(
            lambda: load_dataframe(context_file, config.seldon_data_dir)
        )
        X_context, y_context = _prepare_data(context_df, target_column, feature_columns)
        _validate_params(
            holdout_size, model, len(X_context),
            has_separate_file=predict_file is not None, task_type=task_type,
        )

        if predict_file:
            predict_df = await anyio.to_thread.run_sync(
                lambda: load_dataframe(predict_file, config.seldon_data_dir)
            )
            X_predict = predict_df.to_pandas()
            if target_column in X_predict.columns:
                X_predict = X_predict.drop(columns=[target_column])
            if feature_columns:
                X_predict = X_predict[feature_columns]
        else:
            X_context, X_predict, y_context, _ = train_test_split(
                X_context, y_context, test_size=holdout_size, random_state=random_state,
            )

        seldon = await _make_seldon(ctx, config, model, task_type=task_type)

        def _fit_predict():
            seldon.fit(X_context, y_context)
            preds = seldon.predict(X_predict)
            probas = None
            resolved_task_type = task_type or _get_task_type(seldon)
            if resolved_task_type == "classification" and hasattr(seldon, "predict_proba"):
                try:
                    probas = seldon.predict_proba(X_predict)
                except NotImplementedError:
                    pass
            return preds, probas, resolved_task_type

        predictions, probabilities, resolved_task_type = await anyio.to_thread.run_sync(_fit_predict)
        max_display = 100
        result = {
            "task_type": resolved_task_type,
            "model": model or config.seldon_default_model,
            "num_context_samples": len(X_context),
            "num_predict_samples": len(X_predict),
            "predictions": predictions[:max_display].tolist(),
            "feature_columns": list(X_context.columns),
            "target_column": target_column,
        }

        if len(predictions) > max_display:
            result["note"] = f"Showing first {max_display} of {len(predictions)} predictions"

        if probabilities is not None:
            result["prediction_probabilities"] = probabilities[:max_display].tolist()

        return json.dumps(result, default=str)

    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})
    except NeuralkException as e:
        return json.dumps({"error": f"Neuralk API error: {_sanitize_error(e, config, req_key)}"})
    except Exception as e:
        return json.dumps({"error": f"Prediction failed: {_sanitize_error(e, config, req_key)}"})


@mcp.tool()
async def evaluate(
    ctx: Context,
    context_file: str,
    target_column: str,
    test_file: str | None = None,
    feature_columns: list[str] | None = None,
    model: str | None = None,
    task_type: str | None = None,
    holdout_size: float = 0.2,
    random_state: int = 42,
) -> str:
    """Make predictions and compute performance metrics using Seldon.

    Fits the model with context examples and evaluates predictions against
    ground truth. Returns classification metrics (accuracy, F1, precision,
    recall) or regression metrics (MAE, RMSE, R2) depending on the task.

    Args:
        context_file: Path to the labeled data file used as context examples.
        target_column: Name of the target/label column.
        test_file: Path to labeled test data. If omitted, holds out from context_file.
        feature_columns: Columns to use as features. If omitted, uses all columns except target.
        model: Seldon model variant (seldon-flash, seldon-small, seldon-large). Defaults to server config.
        task_type: Force "classification" or "regression". If omitted, Seldon auto-detects from the target.
        holdout_size: Fraction to hold out for evaluation when test_file is omitted. Defaults to 0.2.
        random_state: Random seed for the holdout split. Defaults to 42.
    """
    config = _get_config(ctx)
    req_key = _get_request_api_key(ctx)

    try:
        context_df = await anyio.to_thread.run_sync(
            lambda: load_dataframe(context_file, config.seldon_data_dir)
        )
        X_context, y_context = _prepare_data(context_df, target_column, feature_columns)
        _validate_params(
            holdout_size, model, len(X_context),
            has_separate_file=test_file is not None, task_type=task_type,
        )

        if test_file:
            test_df = await anyio.to_thread.run_sync(
                lambda: load_dataframe(test_file, config.seldon_data_dir)
            )
            X_test, y_test = _prepare_data(test_df, target_column, feature_columns)
        else:
            X_context, X_test, y_context, y_test = train_test_split(
                X_context, y_context, test_size=holdout_size, random_state=random_state,
            )

        seldon = await _make_seldon(ctx, config, model, task_type=task_type)

        def _fit_predict():
            seldon.fit(X_context, y_context)
            return seldon.predict(X_test), task_type or _get_task_type(seldon)

        predictions, resolved_task_type = await anyio.to_thread.run_sync(_fit_predict)
        if resolved_task_type == "classification":
            metrics = compute_classification_metrics(y_test, predictions)
        else:
            metrics = compute_regression_metrics(y_test, predictions)

        result = {
            "task_type": resolved_task_type,
            "model": model or config.seldon_default_model,
            "num_context_samples": len(X_context),
            "num_test_samples": len(X_test),
            "metrics": metrics,
            "target_column": target_column,
            "feature_columns": list(X_context.columns),
        }

        return json.dumps(result, default=str)

    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})
    except NeuralkException as e:
        return json.dumps({"error": f"Neuralk API error: {_sanitize_error(e, config, req_key)}"})
    except Exception as e:
        return json.dumps({"error": f"Evaluation failed: {_sanitize_error(e, config, req_key)}"})


@mcp.tool()
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
    # Access the lifespan config via a module-level ref set during startup
    config = _lifespan_config
    if config is None:
        return json.dumps({"error": "Server not fully initialized"})
    return json.dumps({
        "default_model": config.seldon_default_model,
        "host": config.neuralk_host or "cloud (api.neuralk-ai.com)",
        "data_dir": config.seldon_data_dir,
        "api_key": (config.neuralk_api_key[:8] + "...") if config.neuralk_api_key else "not set",
    })


# --- MCP Prompts ---


@mcp.prompt()
def classify(file_path: str, target_column: str) -> str:
    """Guide a classification workflow: describe data, predict, and evaluate."""
    return (
        f"I have a classification dataset at '{file_path}' with target column '{target_column}'.\n\n"
        "Please:\n"
        "1. First describe the data using the describe_data tool to understand its structure.\n"
        "2. Then make predictions using the predict tool.\n"
        "3. Evaluate the model using the evaluate tool to see accuracy, F1, and other metrics.\n"
        "4. Summarize the results and suggest if a different model variant might improve performance."
    )


@mcp.prompt()
def regress(file_path: str, target_column: str) -> str:
    """Guide a regression workflow: describe data, predict, and evaluate."""
    return (
        f"I have a regression dataset at '{file_path}' with target column '{target_column}'.\n\n"
        "Please:\n"
        "1. First describe the data using the describe_data tool.\n"
        "2. Make predictions using the predict tool.\n"
        "3. Evaluate the model using the evaluate tool to see MAE, RMSE, and R2 scores.\n"
        "4. Summarize the results."
    )


@mcp.prompt()
def compare_models(file_path: str, target_column: str) -> str:
    """Compare all three Seldon model variants on the same dataset."""
    return (
        f"I want to compare all three Seldon model variants on the dataset at '{file_path}' "
        f"with target column '{target_column}'.\n\n"
        "Please:\n"
        "1. Describe the data first.\n"
        "2. Evaluate seldon-flash, seldon-small, and seldon-large on the same holdout split "
        "(use random_state=42 for consistency).\n"
        "3. Create a comparison table of metrics across all three models.\n"
        "4. Recommend which model to use and why."
    )


# --- CLI ---


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse", "streamable-http"]),
    default="stdio",
    help="MCP transport to use.",
)
@click.option("--port", type=int, default=8000, help="Port for SSE/HTTP transport.")
def main(transport: str, port: int):
    """Start the Seldon MCP server."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "sse":
        mcp.run(transport="sse", port=port)
    else:
        mcp.run(transport="streamable-http", port=port)


if __name__ == "__main__":
    main()
