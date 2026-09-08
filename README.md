# Seldon MCP Server

An [MCP](https://modelcontextprotocol.io) server that gives AI assistants access
to [Seldon](https://www.neuralk-ai.com), Neuralk's tabular foundation model for
classification and regression.

Seldon uses in-context learning: you provide labeled examples as context and it
predicts on new data, with zero hyperparameter tuning. This server is a thin
proxy in front of Neuralk's prediction API. It never reads files from disk: data
reaches Seldon either inline in a tool call (small datasets) or through an
upload straight to Neuralk (any size), and the server only holds the reference.

## Use the hosted server

Neuralk runs this server at **`https://mcp.neuralk.ai/mcp`** (Streamable HTTP).
Nothing to install: point your MCP client at it and send your own Neuralk API
key on every request, as either header:

| Header | Value |
|---|---|
| `x-neuralk-api-key` | `nk_live_...` |
| `Authorization` | `Bearer nk_live_...` |

You are billed on your own account. Create a key at
[prediction.neuralk-ai.com/dashboard/api-keys](https://prediction.neuralk-ai.com/dashboard/api-keys)
or run `neuralk login`.

### Claude Code

```bash
claude mcp add --transport http seldon https://mcp.neuralk.ai/mcp \
  --header "x-neuralk-api-key: nk_live_..."
```

### Cursor, Windsurf, VS Code and other clients with a JSON config

```json
{
  "mcpServers": {
    "seldon": {
      "url": "https://mcp.neuralk.ai/mcp",
      "headers": {
        "x-neuralk-api-key": "nk_live_..."
      }
    }
  }
}
```

### Claude Desktop

Claude Desktop reaches remote servers through a local bridge. Add to
`claude_desktop_config.json` (`~/Library/Application Support/Claude/` on
macOS, `%APPDATA%\Claude\` on Windows):

```json
{
  "mcpServers": {
    "seldon": {
      "command": "npx",
      "args": [
        "-y", "mcp-remote", "https://mcp.neuralk.ai/mcp",
        "--header", "x-neuralk-api-key:${NEURALK_API_KEY}"
      ],
      "env": {
        "NEURALK_API_KEY": "nk_live_..."
      }
    }
  }
}
```

A request without a key is answered `401`; a rejected or revoked key is
reported by the tool call with the reason.

## Tools

| Tool | What it does |
|---|---|
| `predict(dataset_key, label_classes?, max_predictions?)` | Run inference on a dataset already uploaded to Neuralk. The recommended path: no data passes through the server. |
| `predict_from_data(data, target_column, ...)` | Predict from CSV text passed inline. Small datasets only (thousands of rows): the data travels through the tool call. |
| `upload_data(data, target_column, ..., ttl_days?)` | Upload inline CSV once and get a `dataset_id` to predict on many times (e.g. to compare models). |
| `create_upload(num_parts?)` | Start a presigned upload: returns URL(s) to `PUT` a real file to, directly to Neuralk's storage, without the key and without the data passing through the server. The answer is self-describing (archive spec + a ready-to-run recipe). |
| `complete_upload(upload_id, dataset_key, parts)` | Finalize a presigned upload; the `dataset_key` then goes to `predict`. |
| `list_models()` | The three Seldon variants: `seldon-flash` (speed), `seldon-small` (balanced, default), `seldon-large` (accuracy). |

Prediction sets larger than `max_predictions` (default 5000) are truncated
inline and the full CSV is offered as a single-use download link, valid five
minutes.

The `drop_and_predict` prompt walks an assistant through the presigned flow for
a dropped file. Resources: `seldon://models`, `seldon://config`.

**Context selection matters.** Seldon learns from the examples you give it, like
few-shot prompting. Relevant context beats a large one: when predicting for one
customer segment, give examples from that segment, not the whole table.

## Run it yourself

### Locally, over stdio

The classic way: one process per client, the key in the environment.

```bash
export NEURALK_API_KEY=nk_live_...
uvx --from git+https://github.com/Neuralk-AI/mcp-server seldon-mcp
```

Claude Code: `claude mcp add seldon --env NEURALK_API_KEY=nk_live_... -- uvx --from git+https://github.com/Neuralk-AI/mcp-server seldon-mcp`

### As a service, over HTTP

The container serves the Streamable HTTP endpoint at `/mcp` (and `/`), a probe
at `/healthz`, and the download links at `/downloads/<token>`.

```bash
docker build -t seldon-mcp .
docker run --rm -p 8000:8000 -e REQUIRE_CLIENT_API_KEY=true seldon-mcp
```

Every request must then carry the client's key (the hosted mode above). To run
a private instance that bills one organisation instead, set
`NEURALK_API_KEY` and leave `REQUIRE_CLIENT_API_KEY` unset: the server's key is
the fallback for requests that carry none.

`seldon-mcp --transport streamable-http --host 0.0.0.0` runs the same thing
without Docker. Defaults are the hosted ones: stateless (any replica answers any
request), JSON answers (`--sse-response` for event streams),
`--forwarded-allow-ips` for the proxies whose `X-Forwarded-*` headers to trust.

### On Alpic

[Alpic](https://alpic.ai) runs the server serverless from the repository:
[`main.py`](main.py) is the entry point its default `uv run main.py` start
command expects, and [`alpic.json`](alpic.json) pins the install command. Set
`REQUIRE_CLIENT_API_KEY=true`, `SKB_DATA_DIRECTORY=/tmp/skrub_data` and
`MPLCONFIGDIR=/tmp/matplotlib` (the runtime's root filesystem is read-only) in
the environment's variables. Clients send their key as `x-api-key`,
`x-neuralk-api-key` or `Authorization: Bearer`. Two limits of that runtime: a
tool call is cut after 30 seconds, and only `/mcp` is routed, so the
`/downloads/<token>` links for oversized prediction sets are not reachable there.

### On Kubernetes

[`deploy/helm/seldon-mcp`](deploy/helm/seldon-mcp) is the chart, and
[`deploy/README.md`](deploy/README.md) the runbook of the Neuralk deployment
(registry, DNS, certificate, verification).

## Configuration

Environment variables, or a `.env` file next to the process:

| Variable | Default | Description |
|---|---|---|
| `NEURALK_API_KEY` | unset | Server-level key, the fallback when a request carries none. Unset in hosted mode. |
| `REQUIRE_CLIENT_API_KEY` | `false` | Hosted mode: refuse MCP requests without a client key (`401`); never use the server's key on a client's behalf. |
| `SELDON_PUBLIC_URL` | unset | Public base URL the download links are built from (`https://mcp.neuralk.ai`). Empty = each request's own URL. |
| `NEURALK_PREDICTION_URL` | `https://api.prediction.neuralk-ai.com` | The prediction API, which also validates keys (`/api/v1/auth/whoami`). |
| `SELDON_DEFAULT_MODEL` | `seldon-small` | Model when a tool call names none. |
| `SELDON_UPLOAD_TTL_DAYS` | unset (Neuralk default, 90) | Retention of datasets uploaded by `upload_data`: 1, 7, 30 or 90. |
| `SELDON_DOWNLOAD_DIR` | system temp | Where the single-use prediction files are written. |
| `SELDON_DOWNLOAD_TTL_SECONDS` | `300` | How long they live. |
| `SKIP_API_KEY_VALIDATION` | `false` | Skip the upfront key check (not recommended). |
| `API_KEY_VALIDATION_TTL_S` | `300` | Cache of successful key checks. |
| `API_KEY_VALIDATION_TIMEOUT_S` | `5.0` | Timeout of the key check. |

Keys are validated against the auth API before a tool runs; a `401`/`403` comes
back as a clear tool error rather than a traceback mid-inference. If the auth
API is unreachable the check is skipped and the prediction call reports the
real error itself. Keys never appear in logs or error messages.

## Development

```bash
uv sync
uv run ruff check src/ tests/
uv run pytest
```

Tests never reach the network (`pytest-socket`); every outbound call is mocked
at the HTTP boundary. CI also builds the container, boots it and checks that an
MCP request without a key is refused, and lints and renders the Helm chart with
the production values.

## License

Apache 2.0
