from __future__ import annotations

import os
import time
from pathlib import Path

from seldon_mcp.downloads import DownloadStore


class TestDownloadStore:
    def test_save_then_take_single_use(self, tmp_path: Path):
        store = DownloadStore(tmp_path)
        token = store.save(b"hello")
        assert store.take(token) == b"hello"
        # file deleted on serve -> gone the second time
        assert store.take(token) is None
        assert list(tmp_path.glob("*.csv")) == []

    def test_unknown_token(self, tmp_path: Path):
        assert DownloadStore(tmp_path).take("deadbeef") is None

    def test_expired_file_not_served_and_deleted(self, tmp_path: Path):
        store = DownloadStore(tmp_path, ttl_seconds=300)
        token = store.save(b"data")
        # backdate the file's mtime well beyond the TTL
        path = tmp_path / f"{token}.csv"
        old = time.time() - 600
        os.utime(path, (old, old))
        assert store.take(token) is None  # expired
        assert not path.exists()  # and removed

    def test_sweep_removes_only_expired(self, tmp_path: Path):
        store = DownloadStore(tmp_path, ttl_seconds=300)
        fresh = store.save(b"fresh")
        stale = store.save(b"stale")
        stale_path = tmp_path / f"{stale}.csv"
        old = time.time() - 600
        os.utime(stale_path, (old, old))
        removed = store.sweep()
        assert removed == 1
        assert not stale_path.exists()
        assert (tmp_path / f"{fresh}.csv").exists()

    def test_save_sweeps_expired(self, tmp_path: Path):
        store = DownloadStore(tmp_path, ttl_seconds=300)
        stale = store.save(b"stale")
        old = time.time() - 600
        os.utime(tmp_path / f"{stale}.csv", (old, old))
        store.save(b"new")  # save() sweeps first
        assert not (tmp_path / f"{stale}.csv").exists()

    def test_rejects_path_traversal(self, tmp_path: Path):
        store = DownloadStore(tmp_path)
        # non-alphanumeric tokens (path separators, dots) are refused
        assert store.take("../etc/passwd") is None
        assert store.take("a/b") is None
        assert store.take("..") is None

    def test_directory_created(self, tmp_path: Path):
        target = tmp_path / "nested" / "dl"
        DownloadStore(target)
        assert target.is_dir()
