# Seldon MCP Server

An [MCP](https://modelcontextprotocol.io) server that gives LLMs access to [Seldon](https://www.neuralk-ai.com) — Neuralk's tabular foundation model for classification and regression.

Seldon uses in-context learning: you provide labeled examples as context and it predicts on new data, with zero hyperparameter tuning. This server wraps the [neuralk](https://pypi.org/project/neuralk/) Python SDK so any MCP client (Claude Desktop, Claude Code, etc.) can run tabular ML workflows through natural language.

## Quick start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- A Neuralk API key — sign up at [neuralk-ai.com](https://www.neuralk-ai.com) or run `neuralk login`

### Install and run

Run the server directly from GitHub with [uv](https://docs.astral.sh/uv/) — no clone required:

```bash
export NEURALK_API_KEY=nk_live_...
uvx --from git+https://github.com/Neuralk-AI/mcp-server seldon-mcp
```

For local development, clone the repo instead:

```bash
git clone https://github.com/Neuralk-AI/mcp-server.git
cd mcp-server
uv sync
uv run seldon-mcp
```

## Configuration

Set via environment variables or a `.env` file in the project directory:

| Variable | Default | Description |
|---|---|---|
| `NEURALK_API_KEY` | `None` | Server-level API key (optional if clients provide their own via header) |
| `NEURALK_API_BASE_URL` | `https://api.prediction.neuralk-ai.com` | Base URL of the Neuralk SaaS auth API used to validate keys |
| `NEURALK_HOST` | `None` (cloud) | On-premise server URL for the inference SDK |
| `SELDON_DEFAULT_MODEL` | `seldon-small` | Default model variant |
| `SELDON_DATA_DIR` | `.` | Base directory for resolving relative file paths |
| `SKIP_API_KEY_VALIDATION` | `false` | Disable upfront key validation (not recommended) |
| `API_KEY_VALIDATION_TTL_S` | `300` | Cache TTL for successful whoami responses |
| `API_KEY_VALIDATION_TIMEOUT_S` | `5.0` | HTTP timeout for the whoami request |

### API key validation

Before any `predict` or `evaluate` call, the resolved API key is validated
against `GET {NEURALK_API_BASE_URL}/api/v1/auth/whoami`. This catches
revoked / invalid / expired keys with a clear MCP error rather than letting
the SDK fail mid-inference. Successful responses are cached in-process for
`API_KEY_VALIDATION_TTL_S` seconds. If the auth API is unreachable, the
validation step is skipped (fail-open) and the SDK call surfaces the error.

## Connect to an MCP client

### Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "seldon": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/Neuralk-AI/mcp-server", "seldon-mcp"],
      "env": {
        "NEURALK_API_KEY": "nk_live_..."
      }
    }
  }
}
```

### Claude Code

```bash
claude mcp add seldon -- uvx --from git+https://github.com/Neuralk-AI/mcp-server seldon-mcp
```

### Remote (SSE)

```bash
NEURALK_API_KEY=nk_live_... uv run seldon-mcp --transport sse --port 8000
```

Clients connect with their own API key via the `x-neuralk-api-key` header:

```json
{
  "mcpServers": {
    "seldon": {
      "url": "http://your-server:8000/sse",
      "headers": {
        "x-neuralk-api-key": "nk_live_users_own_key"
      }
    }
  }
}
```

**API key resolution order:**
1. `x-neuralk-api-key` HTTP header (per-user)
2. `NEURALK_API_KEY` env var on the server (shared fallback)

If neither is set, predict/evaluate calls return an error. The `describe_data` and `list_models` tools work without a key.

## Tools

### `describe_data`

Load a tabular file and get a statistical summary — shape, column types, null counts, numeric stats, and sample rows.

```
describe_data(file_path="housing.csv")
```

### `predict`

Make predictions with Seldon. Provide a context file with labeled examples and either a separate predict file or use an automatic holdout split.

```
predict(
    context_file="train.csv",
    target_column="price",
    predict_file="new_data.csv",
    model="seldon-large"
)
```

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `context_file` | str | yes | | Path to labeled data (context examples) |
| `target_column` | str | yes | | Name of the target column |
| `predict_file` | str | no | `None` | Data to predict on. If omitted, holds out from context_file |
| `feature_columns` | list[str] | no | all except target | Columns to use as features |
| `model` | str | no | server default | `seldon-flash`, `seldon-small`, or `seldon-large` |
| `holdout_size` | float | no | `0.2` | Holdout fraction (when predict_file is omitted) |
| `random_state` | int | no | `42` | Random seed for the split |

### `evaluate`

Make predictions and compute performance metrics against ground truth.

```
evaluate(
    context_file="data.csv",
    target_column="species",
    model="seldon-small"
)
```

Returns:
- **Classification**: accuracy, F1 (weighted), precision, recall, confusion matrix
- **Regression**: MAE, MSE, RMSE, R², median absolute error

Parameters are the same as `predict`, with `test_file` in place of `predict_file`.

### `list_models`

Returns the available Seldon model variants:

| Model | Description |
|---|---|
| `seldon-flash` | Optimized for low latency |
| `seldon-small` | Balanced speed and accuracy (default) |
| `seldon-large` | Maximum accuracy for complex tasks |

## Resources

| URI | Description |
|---|---|
| `seldon://models` | Available model variants |
| `seldon://config` | Current server configuration (API key masked) |

## Prompts

Pre-built workflow templates for common tasks:

| Prompt | Description |
|---|---|
| `classify(file_path, target_column)` | Guided classification: describe, predict, evaluate |
| `regress(file_path, target_column)` | Guided regression: describe, predict, evaluate |
| `compare_models(file_path, target_column)` | Evaluate all 3 model variants side by side |

## Supported file formats

- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)
- Parquet (`.parquet`)
- JSON (`.json`) — tabular format (array of objects)

## Development

```bash
uv sync
uv run ruff check src/
uv run pytest
```

## License

Apache 2.0
