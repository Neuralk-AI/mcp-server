"""Entry point for MCP hosting platforms that run ``uv run main.py`` (Alpic).

The platform's gateway terminates TLS and authentication in front of this
process and forwards MCP traffic to ``/mcp``. So: Streamable HTTP, bound on all
interfaces, stateless (any instance answers any request), plain JSON answers,
and the platform's proxy headers trusted. The port comes from ``PORT`` when the
platform sets one, else ``FASTMCP_PORT``, else 8000.

For a local or self-hosted run use the ``seldon-mcp`` command instead.
"""

from __future__ import annotations

import os
import sys


def _port() -> int:
    return int(os.environ.get("PORT") or os.environ.get("FASTMCP_PORT") or 8000)


if __name__ == "__main__":
    port = _port()
    # Before the (slow) server import, so the platform's logs show the process
    # is up and which port it is about to bind.
    print(f"seldon-mcp: starting Streamable HTTP on port {port} (python {sys.version.split()[0]})", file=sys.stderr)

    from seldon_mcp.server import serve

    serve(
        transport="streamable-http",
        host="0.0.0.0",
        port=port,
        stateless=True,
        json_response=True,
        forwarded_allow_ips="*",
    )
