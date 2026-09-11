"""Bounded, journaled checkpoint resumes after classified transport failures."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    return json.loads(path.read_text()) if path.exists() else None


def _write(path, value):
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def retry_decision(checkpoint, *, exit_code, checkpoint_fresh, attempts, max_restarts):
    if exit_code == 0:
        if checkpoint and checkpoint.get("status") == "completed":
            return "completed"
        return "incomplete_success_exit"
    if exit_code < 0 or exit_code in {130, 143}:
        return "interrupted"
    if not checkpoint_fresh or not checkpoint or checkpoint.get("status") != "error":
        return "unclassified_exit"
    if checkpoint.get("failure", {}).get("retryable") is not True:
        return "non_retryable_failure"
    return "retry" if attempts <= max_restarts else "restart_limit_reached"


def supervise(config, output, state_dir, *, max_restarts=3, backoff_seconds=30,
              popen=subprocess.Popen, sleeper=time.sleep):
    if isinstance(max_restarts, bool) or not isinstance(max_restarts, int) or not 0 <= max_restarts <= 10:
        raise ValueError("max_restarts must be an integer from 0 through 10")
    if isinstance(backoff_seconds, bool) or not math.isfinite(backoff_seconds) or not 0 <= backoff_seconds <= 3600:
        raise ValueError("backoff_seconds must be finite and between 0 and 3600")
    config, output, state_dir = Path(config).resolve(), Path(output).resolve(), Path(state_dir).resolve()
    if state_dir == output or state_dir.is_relative_to(output):
        raise ValueError("supervisor state must be outside the experiment output")
    manifest = _read(output / "manifest.json")
    if not manifest:
        raise ValueError("supervisor requires an initialized run with a manifest")
    command = [sys.executable, "-m", "moha", "resume", "--config", str(config), "--output", str(output)]
    identity = {"config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "run_identity": manifest["identity_hash"], "command": command,
                "supervisor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "max_restarts": max_restarts, "backoff_seconds": backoff_seconds}
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = state_dir / "state.json"
        state = _read(path)
        if state:
            if state["identity"] != identity:
                raise ValueError("supervisor identity or recovery policy changed")
            if state["status"] in {"completed", "stopped"}:
                return state["exit_code"]
            # An abandoned running child must be investigated, never duplicated.
            if state["status"] == "running":
                raise ValueError("previous child has no recorded exit; inspect it before recovery")
        else:
            state = {"schema": "moha_supervisor_v1", "identity": identity, "attempts": [],
                     "status": "ready", "created_at": _now()}
        state.update(supervisor_pid=os.getpid(), host=socket.gethostname())
        _write(path, state)
        while len(state["attempts"]) <= max_restarts:
            if state["status"] == "waiting_retry":
                delay = min(backoff_seconds * 2 ** (len(state["attempts"]) - 1), 300)
                state.update(retry_delay_seconds=delay, updated_at=_now())
                _write(path, state)
                sleeper(delay)
            attempt = {"number": len(state["attempts"]) + 1, "started_at": _now(),
                       "started_ns": time.time_ns(), "command": command}
            attempt_path = state_dir / f"attempt-{attempt['number']:03d}.json"
            state["attempts"].append(attempt)
            state.update(status="running", updated_at=_now())
            _write(path, state)
            process = None
            try:
                with (state_dir / f"attempt-{attempt['number']:03d}.log").open("ab", buffering=0) as log:
                    process = popen(command, cwd=Path(__file__).resolve().parents[2],
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    attempt["pid"] = process.pid
                    _write(attempt_path, attempt)
                    _write(path, state)
                    code = process.wait()
            except BaseException as exc:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                code = 130 if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 1
                attempt["supervisor_error_type"] = type(exc).__name__
            checkpoint_path = output / "checkpoint.json"
            checkpoint = _read(checkpoint_path)
            fresh = checkpoint_path.exists() and checkpoint_path.stat().st_mtime_ns >= attempt["started_ns"]
            decision = retry_decision(checkpoint, exit_code=code, checkpoint_fresh=fresh,
                                      attempts=len(state["attempts"]), max_restarts=max_restarts)
            if "supervisor_error_type" in attempt:
                decision = "supervisor_interrupted" if code == 130 else "supervisor_error"
            attempt.update(exit_code=code, finished_at=_now(), decision=decision,
                           failure=checkpoint.get("failure") if checkpoint and fresh else None)
            _write(attempt_path, attempt)
            state.update(status="waiting_retry" if decision == "retry" else
                         "completed" if decision == "completed" else "stopped",
                         exit_code=0 if decision == "completed" else code or 1,
                         reason=decision, updated_at=_now())
            _write(path, state)
            if decision != "retry":
                return state["exit_code"]
        raise RuntimeError("supervisor restart accounting is inconsistent")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--backoff-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    def interrupt(*_):
        raise KeyboardInterrupt()
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        return supervise(args.config, args.output, args.state_dir,
                         max_restarts=args.max_restarts, backoff_seconds=args.backoff_seconds)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
