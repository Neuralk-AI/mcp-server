"""Disk-backed, single-use, expiring store for prediction download files.

Predictions that are too large to return inline are written here as CSV and served
once via the server's /downloads route. Cleanup is layered so no single mechanism
is load-bearing:
  - take() deletes the file as it serves it (single use),
  - expired files are never served and are deleted when touched,
  - sweep() (run on a timer + on save + at startup) reclaims never-downloaded files.

Files live on disk (not memory), so they survive request handling and a restart's
leftovers are cleared by the startup sweep.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

DEFAULT_TTL_SECONDS = 300  # 5 minutes


class DownloadStore:
    def __init__(self, directory: str | Path, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self.dir = Path(directory)
        self.ttl_seconds = ttl_seconds
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, data: bytes) -> str:
        """Write the payload to a new token file and return the token."""
        self.sweep()  # opportunistic cleanup on every write
        token = uuid.uuid4().hex
        self.dir.joinpath(f"{token}.csv").write_bytes(data)
        return token

    def take(self, token: str) -> bytes | None:
        """Return the file's bytes and delete it (single use).

        Returns None if the token is unknown, already downloaded, or expired (an
        expired file is removed here too).

        The claim is atomic: the token file is renamed to a unique name before it
        is read, so two concurrent takes on the same token cannot both succeed —
        only the racer whose rename wins gets the bytes (the other sees None).
        """
        path = self._path(token)
        if path is None or not path.exists():
            return None
        if self._is_expired(path):
            path.unlink(missing_ok=True)
            return None
        claim = self.dir / f"{token}.{uuid.uuid4().hex}.claim"
        try:
            path.rename(claim)  # atomic on a single filesystem; only one racer wins
        except OSError:
            return None
        try:
            return claim.read_bytes()
        finally:
            claim.unlink(missing_ok=True)

    def sweep(self) -> int:
        """Delete every expired file. Returns how many were removed."""
        removed = 0
        for path in self.dir.glob("*.csv"):
            if self._is_expired(path):
                path.unlink(missing_ok=True)
                removed += 1
        return removed

    def _is_expired(self, path: Path) -> bool:
        try:
            return (time.time() - path.stat().st_mtime) > self.ttl_seconds
        except OSError:
            return True

    def _path(self, token: str) -> Path | None:
        # Tokens are uuid4 hex; reject anything else so a crafted token cannot
        # escape the download directory (path traversal).
        if not token or not token.isalnum():
            return None
        return self.dir / f"{token}.csv"
