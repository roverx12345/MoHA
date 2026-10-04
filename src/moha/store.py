"""Immutable episode caches, atomic checkpoints and one writer per run."""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
import uuid
from pathlib import Path
from .models import canonical, digest


class RunStore:
    def __init__(self, root: str | Path, identity: dict, *, resume: bool = False):
        self.root = Path(root).resolve()
        self.identity_hash = digest(identity)
        if resume:
            if not self.root.is_dir():
                raise ValueError("resume requires an existing run")
        else:
            self.root.mkdir(parents=True, exist_ok=False)
        self._lock = (self.root / ".lock").open("a+")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = {"schema": "moha_run_v1", "identity": identity, "identity_hash": digest(identity)}
            if resume:
                if self.read("manifest.json") != manifest:
                    raise ValueError("run identity changed; start a new run instead of relabeling cached results")
            else:
                self.write("manifest.json", manifest, immutable=True)
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self._lock.closed:
            self._lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _path(self, name: str) -> Path:
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("artifact path leaves run directory")
        return path

    def read(self, name: str):
        path = self._path(name)
        return json.loads(path.read_text()) if path.exists() else None

    def write(self, name: str, value, *, immutable: bool = False):
        path = self._path(name)
        data = canonical(value) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            if immutable:
                try:
                    os.link(temp, path)
                except FileExistsError:
                    if path.read_text() != data:
                        raise ValueError(f"refusing to replace completed artifact: {name}")
            else:
                os.replace(temp, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def record_error(self, stage: str, payload: dict):
        self.write(f"errors/{stage}-{uuid.uuid4().hex}.json", payload, immutable=True)
