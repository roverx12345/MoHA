"""Launch the deployed Omni server with the experiment's bounded video loader."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

from .bridge import read_object
from .models import positive_int


def media_io_kwargs(config):
    """The loader must retain every frame the host is allowed to send."""
    frames = config["budget"].get("f_view")
    positive_int(frames, "budget.f_view (required for bounded Omni video loading)")
    return {"video": {"num_frames": frames}}


def launch_plan(config, *, vllm_bin, model_path, gpu, host="127.0.0.1",
                endpoint_index=0, max_model_len=32768):
    observer = config["observer"]
    if observer["backend"] != "qwen3omni":
        raise ValueError("this launcher is for the deployed Qwen3-Omni vLLM-Omni stack")
    if not re.fullmatch(r"\d+(,\d+)*", gpu) or len(set(gpu.split(","))) != len(gpu.split(",")):
        raise ValueError("gpu must list distinct available GPU indices, such as 0,1")
    if type(endpoint_index) is not int or endpoint_index < 0:
        raise ValueError("endpoint_index must be a nonnegative integer")
    endpoints = [s.strip().rstrip("/") for s in observer["base_url"].split(",")]
    if endpoint_index >= len(endpoints):
        raise ValueError("endpoint_index is outside observer.base_url")
    endpoint = urlsplit(endpoints[endpoint_index])
    if (endpoint.scheme != "http" or not endpoint.hostname or endpoint.port is None
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment
            or endpoint.path not in ("", "/v1")):
        raise ValueError("a local Omni endpoint requires http, an explicit port, and no credentials")
    positive_int(max_model_len, "max_model_len")
    loader = media_io_kwargs(config)
    command = [str(vllm_bin), "serve", str(model_path), "--omni", "--host", host,
               "--port", str(endpoint.port), "--max-model-len", str(max_model_len),
               "--enforce-eager", "--served-model-name", observer["model"],
               "--media-io-kwargs", json.dumps(loader, separators=(",", ":"))]
    return {"command": command, "environment": {"CUDA_VISIBLE_DEVICES": gpu,
            "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"},
            "observer_endpoint": endpoints[endpoint_index], "media_io_kwargs": loader,
            "host_max_frames": config["budget"]["f_view"],
            "max_model_len": max_model_len, "server_verified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu", required=True, help="Explicitly selected, available GPU indices")
    parser.add_argument("--vllm-bin", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--endpoint-index", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--launch-record", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = read_object(args.config)
    plan = launch_plan(config, vllm_bin=args.vllm_bin, model_path=args.model_path,
        gpu=args.gpu, host=args.host, endpoint_index=args.endpoint_index,
        max_model_len=args.max_model_len)
    plan.update(config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                config_path=str(args.config.resolve()))
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if args.launch_record is None:
        parser.error("--launch-record is required outside --dry-run")
    record = args.launch_record.resolve()
    roots = [Path(__file__).resolve().parents[2], Path(config["runtime"]["root"]).resolve()]
    if any(record.is_relative_to(root) for root in roots):
        raise ValueError("launch record must be outside source and pinned runtime")
    if not args.vllm_bin.is_file() or not os.access(args.vllm_bin, os.X_OK):
        raise ValueError("vllm-bin must be an executable file")
    if not args.model_path.is_dir():
        raise ValueError("model-path must be an existing model directory")
    record.parent.mkdir(parents=True, exist_ok=True)
    plan.update(started_at=datetime.now(timezone.utc).isoformat(), pid=os.getpid())
    with record.open("x") as output:
        json.dump(plan, output, indent=2)
        output.write("\n")
    # No free-form shell arguments, credentials, process termination or restart.
    os.execvpe(plan["command"][0], plan["command"], {**os.environ, **plan["environment"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
