"""python -m moha: doctor, run, resume, evaluate, demo."""
from __future__ import annotations
import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from .models import Harness, check_splits, digest
from .store import RunStore


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paper-aligned MOHA with a fixed Video OS runtime")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "run", "resume", "evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True, type=Path)
        if name != "doctor":
            sub.add_argument("--output", required=True, type=Path)
        if name == "evaluate":
            sub.add_argument("--frozen", required=True, type=Path)
            sub.add_argument("--manifest", required=True, type=Path)
            sub.add_argument("--resume", action="store_true")
    demo = commands.add_parser("demo")
    demo.add_argument("--output", required=True, type=Path)
    demo.add_argument("--resume", action="store_true")
    export = commands.add_parser("export", help="Record a completed planner/observer calibration without credentials")
    export.add_argument("--run", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "export":
        from .profiles import export_profile
        result = export_profile(args.run, args.output)
    elif args.command == "demo":
        from .demo import run_demo
        result = run_demo(args.output, args.resume)
    else:
        from .bridge import build, load_split, prepare, read_object
        repo = Path(__file__).resolve().parents[2]
        prepared = prepare(args.config, repo)
        if args.command == "doctor":
            from .serving import media_io_kwargs
            result = {"inputs_valid": True, "clean_worktree": not prepared["identity"]["source"]["dirty"],
                      "clean_runtime": not prepared["identity"]["runtime"]["dirty"],
                      "runtime_commit": prepared["identity"]["runtime"]["commit"],
                      "calibration_samples": len(prepared["calibration"]),
                      "validation_samples": len(prepared["validation"]),
                      "execution_lanes": list(prepared["observer_endpoints"]),
                      "planner_endpoints": list(prepared["planner_endpoints"]),
                      "diagnosis_workers": len(prepared["clients"]["judges"]),
                      "observer_media_loading": {
                          "required_media_io_kwargs": media_io_kwargs(prepared["config"])
                              if prepared["budget"].f_view is not None else None,
                          "server_verified": False},
                      "available_catalog": prepared["allowed_ids"], "model_calls": 0,
                      "configured_specialists": {
                          "ocr": prepared["config"]["image"]["spec"]["model"] if "ocr" in prepared["config"].get("specialists", []) else None,
                          "asr": prepared["config"]["asr"]["spec"]["model"] if "asr" in prepared["config"].get("specialists", []) else None},
                      "source_hash": prepared["identity"]["source"]["source_hash"],
                      "note": "Configuration, imports, credential availability and video hashes checked; endpoint health is not measured."}
        else:
            if prepared["identity"]["source"]["dirty"] or prepared["identity"]["runtime"]["dirty"]:
                raise ValueError("AGENTS.md requires a clean committed worktree before evaluation/inference; doctor and offline tests remain available")
            if any(args.output.resolve().is_relative_to(root) for root in (
                    repo, Path(prepared["config"]["runtime"]["root"]).resolve())):
                raise ValueError("run output must be outside the source worktree")
            identity = prepared["identity"]
            if args.command == "evaluate":
                frozen = read_object(args.frozen)
                if frozen.get("schema") != "moha_frozen_v1" or frozen.get("run_identity") != digest(identity):
                    raise ValueError("frozen harness provenance differs from the current source/model/input configuration")
                harness = Harness.from_dict(frozen["harness"])
                if harness.id != frozen.get("harness_id"):
                    raise ValueError("frozen harness hash mismatch")
                samples, assets, manifest = load_split(args.manifest, prepared["config"]["media_root"], "test")
                check_splits(prepared["calibration"] + prepared["validation"], samples)
                if prepared["media_hashes"] & {s["media_sha256"] for s in manifest["samples"]}:
                    raise ValueError("test video content overlaps calibration/validation")
                prepared["assets"].update(assets)
                identity = {**identity, "evaluation": {"frozen": frozen, "input_hash": digest(manifest)}}
            with RunStore(args.output, identity, resume=args.command == "resume" or getattr(args, "resume", False)) as store:
                store.write(f"invocations/{uuid.uuid4().hex}.json", {
                    "command": sys.argv if argv is None else argv, "cwd": str(Path.cwd()),
                    "started_at": datetime.now(timezone.utc).isoformat(), "output": str(store.root)}, immutable=True)
                calibrator = build(prepared, store)
                if args.command == "evaluate":
                    episodes = calibrator.batch(harness, samples, 0, "test")
                    result = {"status": "completed", "harness_id": harness.id, "samples": len(episodes),
                              "correct": sum(e.correct for e in episodes),
                              "accuracy": sum(e.correct for e in episodes) / len(episodes)}
                    store.write("evaluation.json", result, immutable=True)
                else:
                    result = calibrator.run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        # Provider exception strings are not printed at this boundary.
        print(f"MOHA failed: {type(exc).__name__}: {exc}" if isinstance(exc, ValueError)
              else f"MOHA failed: {type(exc).__name__}; inspect run artifacts", file=sys.stderr)
        raise SystemExit(1)
