"""Credential-free registry records from completed calibration artifacts."""
from __future__ import annotations
import json
import os
import re
import tempfile
from pathlib import Path
from .catalog import Intervention
from .models import Harness, ValidationPolicy, digest


def calibrated_profile(run):
    run = Path(run).resolve()
    names = ("manifest.json", "experiment.json", "checkpoint.json", "result.json", "frozen_harness.json")
    missing = [name for name in names if not (run / name).is_file()]
    if missing:
        raise ValueError("calibration incomplete: missing " + ", ".join(missing))
    records = {name: json.loads((run / name).read_text()) for name in names}
    manifest, experiment, checkpoint, result, frozen = (records[n] for n in names)
    if manifest.get("schema") != "moha_run_v1" or frozen.get("schema") != "moha_frozen_v1":
        raise ValueError("export requires native MoHA calibration artifacts")
    identity = manifest["identity"]
    if digest(identity) != manifest["identity_hash"] or frozen["run_identity"] != manifest["identity_hash"]:
        raise ValueError("calibration run identity mismatch")
    if result.get("status") != "completed" or checkpoint != result:
        raise ValueError("only a completed, consistent calibration can be exported")
    if not result.get("stop_reason") or identity.get("source", {}).get("dirty", True):
        raise ValueError("export requires a committed source and an explicit stop reason")
    if identity.get("runtime", {}).get("dirty", False):
        raise ValueError("export requires a committed runtime")
    current = Harness.from_dict(experiment["initial"])
    catalog = {item["id"]: Intervention(**item) for item in experiment["catalog"]}
    decisions = []
    for row in result["history"]:
        proposed = catalog[row["candidate"]].apply(current)
        if row["from"] != current.id or row["to"] != proposed.id or type(row["validation"]["accepted"]) is not bool:
            raise ValueError("calibration acceptance history is inconsistent")
        accepted = row["validation"]["accepted"]
        decisions.append({"candidate": row["candidate"], "accepted": accepted,
                          "from": row["from"], "to": row["to"]})
        if accepted:
            current = proposed
    perception = None
    if experiment.get("perception_calibration", False):
        from .perception import policy_grid, validate_result
        for name in ("perception/plan.json", "perception/result.json"):
            if not (run / name).is_file():
                raise ValueError("calibration incomplete: missing " + name)
            records[name] = json.loads((run / name).read_text())
        plan, perception = records["perception/plan.json"], records["perception/result.json"]
        if (plan.get("structural_harness") != current.to_dict() or plan.get("configs") != policy_grid(current)
                or plan.get("validation_samples") != experiment["heldout"]
                or plan.get("validation_policy") != experiment["validation"]
                or result.get("structural_harness") != current.to_dict()
                or result.get("perception_calibration") != perception):
            raise ValueError("final perception calibration differs from the structural result or validation contract")
        current = validate_result(perception, current, ValidationPolicy(**experiment["validation"]))
        for row in perception["rows"]:
            name = f"perception/scores/{row['id']}.json"
            if not (run / name).is_file() or json.loads((run / name).read_text()) != row:
                raise ValueError("final perception scores are incomplete or inconsistent")
            records[name] = row
    if current.id != frozen["harness_id"] or current != Harness.from_dict(frozen["harness"]) or current != Harness.from_dict(result["harness"]):
        raise ValueError("frozen harness differs from the calibration result/history")
    config = identity["config"]
    # Only fixed, non-credential provider settings leave the experiment directory.
    spec_fields = {"model", "base_url", "response_format_mode", "timeout_seconds", "retries",
                   "max_completion_tokens", "temperature", "seed", "reasoning_effort", "enable_thinking",
                   "thinking_wire_format", "reasoning_preamble_mode"}
    def spec(value):
        return {k: v for k, v in value["spec"].items() if k in spec_fields}
    stack = {"models": {name: spec(value) for name, value in config["models"].items()},
             "observer": {k: v for k, v in config["observer"].items()
                          if k in {"backend", "model", "base_url", "timeout_seconds", "retries"}}}
    for name in ("image", "asr"):
        if name in config:
            stack[name] = spec(config[name])
            if name == "asr":
                stack[name]["language"] = config[name].get("language")
    pair = {"planner": stack["models"]["planner"]["model"], "observer": stack["observer"]["model"]}
    source_fields = {"commit", "source_hash"}
    return {"schema": "moha_calibrated_profile_v1", "pair": pair, "stack": stack,
        "budget": config["budget"], "harness": current.to_dict(), "harness_id": current.id,
        "calibration": {"stop_reason": result["stop_reason"], "decisions": decisions,
                        "search": experiment["search"], "validation": experiment["validation"],
                        "perception_selection": perception},
        "provenance": {"run": str(run), "run_identity": manifest["identity_hash"],
            "source": {k: v for k, v in identity["source"].items() if k in source_fields},
            "runtime": {k: v for k, v in identity.get("runtime", {}).items() if k in source_fields},
            "input_hashes": identity["input_hashes"], "artifacts": {k: digest(v) for k, v in records.items()}}}


def export_profile(run, output):
    profile = calibrated_profile(run)
    output = Path(output).resolve()
    run = Path(run).resolve()
    if output.is_relative_to(run):
        raise ValueError("profile registry must be outside the completed run")
    output.mkdir(parents=True, exist_ok=True)
    pair = profile["pair"]
    slug = lambda value: re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-.")[:80] or "model"
    name = f"{slug(pair['planner'])}__{slug(pair['observer'])}__{digest(pair)[:8]}.json"
    target = output / name
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=output)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(profile, ensure_ascii=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"profile": str(target), "pair": pair, "harness_id": profile["harness_id"]}
