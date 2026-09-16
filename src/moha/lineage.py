"""Audited continuation of a paused shared run under endpoint-only changes."""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path

from .models import Harness, digest


LINEAGE_SCHEMA = "moha_shared_lineage_v1"
IMPORT_POLICY = "completed_episode_diagnosis_probe_and_coordinator_state_v1"
SAFE_SOURCE_CHANGES = frozenset({
    "src/moha/__main__.py",
    "src/moha/bridge.py",
    "src/moha/lineage.py",
    "src/moha/shared.py",
})
SHARED_ARTIFACT_DIRS = ("recommendations", "profiles", "selections")
STACK_ARTIFACT_DIRS = ("episodes", "diagnoses", "probes")


def _file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _without_endpoint_placement(identity: dict) -> dict:
    """Retain experiment semantics while excluding audited infrastructure placement."""
    normalized = copy.deepcopy(identity)
    normalized.pop("lineage", None)
    for stack in normalized.get("stacks", []):
        item = stack["identity"]
        item["source"] = "validated-separately"
        config = item["config"]
        config.pop("planner_max_inflight_per_endpoint", None)
        config["observer"]["base_url"] = "audited-endpoint-pool"
        config["models"]["planner"]["spec"]["base_url"] = "audited-endpoint-pool"
    return normalized


def _source_delta(parent_identity: dict, child_identity: dict) -> list[dict]:
    changes = None
    result = []
    for parent_stack, child_stack in zip(parent_identity["stacks"], child_identity["stacks"]):
        parent = parent_stack["identity"]["source"]
        child = child_stack["identity"]["source"]
        if parent.get("dirty") or child.get("dirty"):
            raise ValueError("shared fork requires clean parent and child source snapshots")
        old_files, new_files = parent.get("files", {}), child.get("files", {})
        current = sorted(path for path in set(old_files) | set(new_files)
                         if old_files.get(path) != new_files.get(path))
        if set(current) - SAFE_SOURCE_CHANGES:
            raise ValueError("shared fork changes calibration semantics outside the approved lineage boundary")
        parent_environment = {key: value for key, value in parent.items()
                              if key not in {"commit", "dirty", "source_hash", "files"}}
        child_environment = {key: value for key, value in child.items()
                             if key not in {"commit", "dirty", "source_hash", "files"}}
        if parent_environment != child_environment:
            raise ValueError("shared fork source environment changed")
        if changes is None:
            changes = current
        elif changes != current:
            raise ValueError("shared fork stacks do not share one source transition")
        result.append({"stack_id": parent_stack["stack_id"],
                       "parent_commit": parent.get("commit"),
                       "child_commit": child.get("commit")})
    return [{"files": changes or [], "commits": result}]


def _validate_identities(parent_identity: dict, child_identity: dict) -> list[dict]:
    if parent_identity.get("implementation") != "moha_shared_calibration_v1":
        raise ValueError("parent is not a shared calibration run")
    if parent_identity.get("stack_order") != child_identity.get("stack_order"):
        raise ValueError("shared fork stack order changed")
    if _without_endpoint_placement(parent_identity) != _without_endpoint_placement(child_identity):
        raise ValueError("shared fork permits only source, endpoint-pool and planner-concurrency changes")
    return _source_delta(parent_identity, child_identity)


def _artifact_paths(parent_root: Path, stack_ids: list[str]) -> list[Path]:
    roots = [parent_root / name for name in SHARED_ARTIFACT_DIRS]
    roots.extend(parent_root / "models" / stack_id / name
                 for stack_id in stack_ids for name in STACK_ARTIFACT_DIRS)
    paths = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"invalid parent artifact directory: {root}")
        for path in sorted(root.rglob("*")):
            if path.is_symlink() or not path.is_file():
                if path.is_symlink():
                    raise ValueError(f"parent artifact must not be a symlink: {path}")
                continue
            paths.append(path)
    return paths


def plan_shared_fork(parent_store, child_identity: dict) -> dict:
    """Validate a locked parent and bind the child identity to exact imported bytes."""
    manifest = parent_store.read("manifest.json")
    if not isinstance(manifest, dict) or manifest.get("identity_hash") != parent_store.identity_hash:
        raise ValueError("parent manifest identity is invalid")
    parent_identity = manifest.get("identity")
    if not isinstance(parent_identity, dict):
        raise ValueError("parent manifest lacks a shared identity")
    source_transition = _validate_identities(parent_identity, child_identity)
    checkpoint = parent_store.read("checkpoint.json")
    if not isinstance(checkpoint, dict) or checkpoint.get("status") == "completed":
        raise ValueError("shared fork requires an unfinished parent checkpoint")
    if checkpoint.get("phase") is not None:
        raise ValueError("shared fork is supported only before final perception calibration")
    round_id = checkpoint.get("round")
    if type(round_id) is not int or round_id < 0:
        raise ValueError("parent checkpoint has an invalid round")
    for index in range(round_id):
        for directory in ("recommendations", "profiles"):
            if parent_store.read(f"{directory}/{index:03d}.json") is None:
                raise ValueError(f"parent checkpoint is missing {directory} round {index}")
    paths = _artifact_paths(parent_store.root, list(parent_identity["stack_order"]))
    records = []
    for path in paths:
        stat = path.stat()
        records.append({"path": str(path.relative_to(parent_store.root)),
                        "bytes": stat.st_size, "sha256": _file_hash(path)})
    descriptor = {
        "schema": LINEAGE_SCHEMA,
        "parent_identity_hash": parent_store.identity_hash,
        "parent_checkpoint_hash": digest(checkpoint),
        "parent_round": round_id,
        "parent_attempt": checkpoint.get("attempt"),
        "parent_harness_id": Harness.from_dict(checkpoint["harness"]).id,
        "import_policy": IMPORT_POLICY,
        "artifact_index_hash": digest(records),
        "artifact_count": len(records),
        "artifact_bytes": sum(record["bytes"] for record in records),
        "source_transition": source_transition,
    }
    return {"descriptor": descriptor, "artifacts": records,
            "checkpoint": checkpoint, "parent_root": str(parent_store.root)}


def import_shared_fork(parent_store, child_store, plan: dict) -> dict:
    """Hard-link immutable caches and copy the mutable coordinator checkpoint."""
    child_manifest = child_store.read("manifest.json")
    if child_manifest.get("identity", {}).get("lineage") != plan["descriptor"]:
        raise ValueError("child run identity is not bound to the fork plan")
    imported = []
    for record in plan["artifacts"]:
        source = parent_store._path(record["path"])
        if source.stat().st_size != record["bytes"] or _file_hash(source) != record["sha256"]:
            raise ValueError(f"parent artifact changed while locked: {record['path']}")
        target = child_store._path(record["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError(f"child artifact already exists: {record['path']}")
        os.link(source, target)
        imported.append(record["path"])
    child_store.write("checkpoint.json", plan["checkpoint"])
    audit = {**plan["descriptor"], "parent_root": plan["parent_root"],
             "artifacts": plan["artifacts"], "link_method": "hardlink",
             "checkpoint_copy": "independent_atomic_copy"}
    child_store.write("lineage.json", audit, immutable=True)
    return {"status": "forked", "run_identity": child_store.identity_hash,
            "parent_identity": parent_store.identity_hash,
            "round": plan["descriptor"]["parent_round"],
            "harness_id": plan["descriptor"]["parent_harness_id"],
            "imported_artifacts": len(imported),
            "imported_bytes": plan["descriptor"]["artifact_bytes"],
            "model_calls": 0}
