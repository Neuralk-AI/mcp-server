# syntax=docker/dockerfile:1

# The hosted Seldon MCP server: one process serving the Streamable HTTP MCP
# endpoint. Built by .github/workflows/image.yml, run by deploy/helm/seldon-mcp.
#
#   docker build -t seldon-mcp .
#   docker run --rm -p 8000:8000 -e REQUIRE_CLIENT_API_KEY=true seldon-mcp

# --- build stage: resolve the locked dependencies into a virtualenv -----------
FROM python:3.11-slim AS builder

# uv, pinned: the same resolver the repository is developed with, so the image
# carries exactly what uv.lock says and nothing newer.
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
# pyproject names README.md as the package readme; the build needs it present.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# --- runtime stage ------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"]

# The binary is the entrypoint, the flags are the default arguments: a runtime
# that passes its own arguments (Kubernetes `args`, `docker run image --flag`)
# replaces the flags and keeps the binary. Stateless Streamable HTTP, JSON
# answers, X-Forwarded-* trusted: the settings for running behind an ingress.
ENTRYPOINT ["seldon-mcp"]
CMD ["--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8000", \
     "--stateless", "--json-response", "--forwarded-allow-ips", "*"]
