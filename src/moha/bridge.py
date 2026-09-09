"""All environment/provider/Video OS dependencies live at this boundary."""
from __future__ import annotations
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit
from .catalog import catalog
from .loop import Calibrator, SearchPolicy
from .models import Harness, Sample, ValidationPolicy, canonical, check_splits, digest, positive_int
from .probes import ObserverResolver, ProbeRunner
from .roles import Judge
from .runtime import EpisodeRunner
from .execution import FRAME_CAP


def read_object(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("configuration/manifest must be a JSON object")
    canonical(value)
    return value


def credential(reference):
    from video_os.core.credentials import load_gpt_credentials
    if set(reference) == {"env"}:
        key = os.environ.get(reference["env"], "")
        if not key.strip():
            raise ValueError(f"required credential environment variable is unset: {reference['env']}")
        return key.strip()
    if set(reference) == {"file", "field"}:
        return load_gpt_credentials(reference["file"], gpt_field=reference["field"]).gpt_api_key
    if reference == {"local": True}:
        return "local-service"
    raise ValueError("key must reference env, file+field, or local=true; inline secrets are unsupported")


def text_client(config, budget, *, structured=False):
    from video_os.core.dispatch import ProviderRole
    from video_os.providers.client import OpenAICompatibleAdapter, ProviderSpec
    from video_os.agent.planner import OpenAICompatiblePlannerClient
    if set(config) != {"spec", "key"}:
        raise ValueError("text model configuration requires spec and key only")
    if "role" in config["spec"]:
        raise ValueError("provider role is fixed by the runtime")
    spec = ProviderSpec(role=ProviderRole.GPT_TEXT, **{
        "response_format_mode": "json_schema" if structured else "json_text", "retries": 0,
        **config["spec"]})
    if spec.seed is not None and not structured:
        raise ValueError("the shared planner adapter does not transmit seed; omit it rather than claim deterministic repeats")
    client_type = OpenAICompatibleAdapter if structured else OpenAICompatiblePlannerClient
    return client_type(spec=spec, api_key=credential(config["key"]), budget=budget)


def whisper_factory(config):
    """Reuse the pinned runtime's bounded-audio transcription adapter."""
    from video_os.core.dispatch import ProviderRole
    from video_os.providers.client import ProviderSpec, WhisperTranscriptionAdapter
    if set(config) - {"spec", "key", "language"} or not {"spec", "key"} <= set(config):
        raise ValueError("ASR configuration requires spec, key, and optional language")
    allowed = {"model", "base_url", "timeout_seconds", "retries"}
    if set(config["spec"]) - allowed or not {"model", "base_url"} <= set(config["spec"]):
        raise ValueError("ASR spec requires model/base_url and optional timeout_seconds/retries")
    language = config.get("language")
    if language is not None and (not isinstance(language, str) or not language.strip()):
        raise ValueError("ASR language must be a nonempty language code or null for automatic detection")
    spec = ProviderSpec(role=ProviderRole.MULTIMODAL_SENSOR, response_format_mode="json_text",
        max_completion_tokens=None, require_usage=False, **{"retries": 0, **config["spec"]})
    key = credential(config["key"])

    def create(artifact_root, schemas, budget):
        return WhisperTranscriptionAdapter(spec=spec, api_key=key, budget=budget,
            artifact_root=artifact_root, schemas=schemas, default_language=language)
    return create


class TextRoleClient:
    def __init__(self, adapter):
        self.adapter = adapter

    def call(self, prompt, payload, schema, name):
        from video_os.core.context import ConservativeTokenCounter
        from video_os.core.dispatch import Mode, ProviderRole, RequestEnvelope, sanitize_gpt_text_payload
        from video_os.core.errors import ProviderResponseError
        from video_os.providers.client import parse_provider_json_object
        safe = sanitize_gpt_text_payload(payload)
        request = RequestEnvelope(role=ProviderRole.GPT_TEXT, mode=Mode.THINK,
            control_prompt=prompt, payload=safe,
            estimated_input_tokens=ConservativeTokenCounter().count(prompt + "\n" + canonical(safe)))
        try:
            response = self.adapter.call(request, output_schema=schema, schema_name=name)
            return response.parsed, {**response.audit_dict(), "raw_text": response.raw_text}
        except ProviderResponseError as exc:
            receipt = exc.receipt
            if receipt is None:
                raise
            exc.moha_audit = receipt.audit_dict()
            # Missing usage is an audit/infrastructure failure, not a format repair.
            if self.adapter.spec.require_usage and receipt.usage.input_tokens is None:
                raise
            ceiling = self.adapter.budget.c_text_max
            if ceiling is not None and receipt.usage.input_tokens is not None and receipt.usage.input_tokens > ceiling:
                raise
            try:
                value = parse_provider_json_object(receipt.raw_text)
            except ProviderResponseError:
                value = receipt.raw_text
            return value, {**receipt.audit_dict(), "raw_text": receipt.raw_text, "wire_schema_valid": False}


def load_split(path, media_root, role):
    from run_eval import validate_input_manifest
    raw = read_object(path)
    if raw.get("selection_split") not in (None, role):
        raise ValueError(f"manifest split differs from requested {role}")
    normalized, assets = validate_input_manifest(raw, media_root=Path(media_root))
    metadata = {s["sample_id"]: s for s in raw["samples"]}
    samples = []
    for row in normalized["samples"]:
        original = metadata[row["sample_id"]]
        samples.append(Sample(row["sample_id"], original.get("video_id"),
            original.get("dataset") or raw.get("dataset"), row["asset_id"], row["task"], row["expected_answer"]))
    return samples, assets, normalized


def _snapshot(repo, directories, names):
    repo = Path(repo).resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    files = {}
    for directory in directories:
        for path in sorted((repo / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json", ".md", ".txt", ".sh"} and "__pycache__" not in path.parts:
                files[str(path.relative_to(repo))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in names:
        path = repo / name
        if path.is_file():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    packages = {}
    for name in ("numpy", "Pillow", "av", "requests", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "source_hash": digest(files), "files": files,
            "environment": {"python": sys.version, "executable": sys.executable,
                            "platform": platform.platform(), "packages": packages}}


def source_identity(repo):
    return _snapshot(repo, ("src/moha", "scripts"), ("pyproject.toml", "AGENTS.md", "README.md", "config.example.json"))


def activate_runtime(reference):
    """Bind to one explicit, committed Video OS dependency. Never search versions."""
    if not isinstance(reference, dict) or set(reference) != {"root", "commit"}:
        raise ValueError("runtime requires exactly root and commit")
    root = Path(reference["root"]).expanduser().resolve(strict=True)
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != reference["commit"]:
        raise ValueError("Video OS runtime commit differs from configured pin")
    if str(root) not in sys.path:
        sys.path.append(str(root))
    for name in ("video_os", "observer_harness", "run_eval"):
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.origin or not Path(spec.origin).resolve().is_relative_to(root):
            raise ValueError(f"{name} import does not come from the pinned runtime")
    return root


def lane_clients(models, observer_url, budget):
    """Pair ordered endpoint pools, broadcasting a shared provider to each lane."""
    from video_os.providers.scheduler import normalize_endpoint_urls
    planners = normalize_endpoint_urls(models["planner"]["spec"]["base_url"])
    observers = normalize_endpoint_urls(observer_url)
    count = max(len(planners), len(observers))
    if any(len(pool) not in (1, count) for pool in (planners, observers)):
        raise ValueError("planner/observer endpoint counts must match or one must be a singleton")
    planners = planners * count if len(planners) == 1 else planners
    observers = observers * count if len(observers) == 1 else observers
    lanes = []
    for endpoint in planners:
        planner = models["planner"]
        lane_models = {**models, "planner": {**planner, "spec": {**planner["spec"], "base_url": endpoint}}}
        lanes.append({name: text_client(value, budget) for name, value in lane_models.items() if name != "judge"})
    return planners, observers, lanes


def prepare(config_path, repo):
    config = read_object(config_path)
    required = {"schema", "runtime", "media_root", "calibration_manifest", "validation_manifest", "budget", "models", "observer"}
    allowed = required | {"initial", "search", "validation", "specialists", "image", "asr", "retrieval_extension", "diagnosis_workers", "perception_calibration"}
    if required - set(config) or set(config) - allowed or config["schema"] != "moha_config_v1":
        raise ValueError("unknown or missing MOHA configuration fields/schema")
    judge_spec = config["models"].get("judge", {}).get("spec", {})
    if judge_spec.get("model", "").strip().lower() == "gpt-5.5" and any(
            (urlsplit(endpoint.strip()).hostname or "").startswith("216.")
            for endpoint in judge_spec.get("base_url", "").split(",")):
        raise ValueError("216.* gpt-5.5 Judge is retired; configure models.judge with "
                         "claude-opus-4-8 at https://zgc.apihy.com, json_text, and Claude credentials "
                         "as in config.example.json. Use a new run output; unchanged existing runs "
                         "must resume with their original frozen source/configuration.")
    diagnosis_workers = config.get("diagnosis_workers", 8)
    positive_int(diagnosis_workers, "diagnosis_workers")
    runtime_root = activate_runtime(config["runtime"])
    from video_os.core.budget import BudgetContract
    initial = Harness.from_dict(config.get("initial", {}))
    # Runtime controls may be adjusted consistently across stacks, while all
    # adaptations must begin with the common two-tool, single-Omni H0.
    if any((initial.overview, initial.memory, initial.verification, initial.retrieval_guard, initial.specialists)) or initial.execution != Harness().execution:
        raise ValueError("calibration must start from shared Initial-Omni H0")
    search, validation = SearchPolicy(**config.get("search", {})), ValidationPolicy(**config.get("validation", {}))
    budget = BudgetContract(**config["budget"])
    perception_calibration = config.get("perception_calibration", False)
    if type(perception_calibration) is not bool:
        raise ValueError("perception_calibration must be boolean")
    if perception_calibration and budget.f_view != FRAME_CAP:
        raise ValueError("final perception calibration requires the shared f_view=128 frame cap")
    specialists = config.get("specialists", [])
    if not isinstance(specialists, list) or len(set(specialists)) != len(specialists) or set(specialists) - {"ocr", "asr"}:
        raise ValueError("specialists must list supported, unique capabilities")
    if "ocr" in specialists and "image" not in config:
        raise ValueError("OCR requires an explicitly configured image backend")
    if "asr" in specialists and "asr" not in config:
        raise ValueError("ASR requires an explicitly configured Whisper backend")
    observer_keys = {"backend", "model", "base_url", "key", "timeout_seconds", "retries"}
    if set(config["observer"]) - observer_keys or not {"backend", "model", "base_url", "key"} <= set(config["observer"]):
        raise ValueError("unknown or missing observer configuration fields")
    if config["observer"]["backend"] not in {"qwen3omni", "qwen2.5omni"}:
        raise ValueError("this MOHA implementation supports the deployed Qwen Omni services")
    if type(config.get("retrieval_extension", False)) is not bool:
        raise ValueError("retrieval_extension must be boolean")
    models = config["models"]
    if not {"planner", "judge"} <= set(models) or set(models) - {"planner", "judge", "extractor"}:
        raise ValueError("models require planner/judge and optionally extractor; selection is deterministic")
    planner_endpoints, endpoints, lanes = lane_clients(models, config["observer"]["base_url"], budget)
    clients = {"judges": [text_client(models["judge"], budget, structured=True) for _ in range(diagnosis_workers)],
               "probe_judges": [text_client(models["judge"], budget, structured=True) for _ in lanes],
               "lanes": lanes}
    credential(config["observer"]["key"])
    if "image" in config:
        allowed_image = {"model", "base_url", "retries", "timeout_seconds", "max_completion_tokens", "enable_thinking"}
        if set(config["image"].get("spec", {})) - allowed_image:
            raise ValueError("image spec contains fields the shared image service cannot apply")
        text_client(config["image"], budget)  # Validate the endpoint/key without calling it.
    if "asr" in config:
        whisper_factory(config["asr"])  # Validate the transcription contract without calling it.
    calibration, assets, left = load_split(config["calibration_manifest"], config["media_root"], "calibration")
    validation_samples, right_assets, right = load_split(config["validation_manifest"], config["media_root"], "validation")
    check_splits(calibration, validation_samples)
    if {s["media_sha256"] for s in left["samples"]} & {s["media_sha256"] for s in right["samples"]}:
        raise ValueError("identical video content occurs in both splits")
    source = source_identity(repo)
    runtime = _snapshot(runtime_root, ("video_os", "observer_harness", "schemas", "prompts"), ("run_eval.py", "AGENTS.md"))
    identity = {"implementation": "moha", "config": config, "source": source, "runtime": runtime,
                "input_hashes": {"calibration": digest(left), "validation": digest(right)}}
    ids = [k for k, item in catalog().items()
           if (item.coordinate != "specialists" or item.value in specialists)
           and not (perception_calibration and item.coordinate.startswith("execution."))
           and (item.coordinate != "retrieval_guard" or config.get("retrieval_extension", False))]
    return {"config": config, "identity": identity, "initial": initial, "search": search,
            "policy": validation, "budget": budget, "assets": {**assets, **right_assets},
            "calibration": calibration, "validation": validation_samples, "clients": clients,
            "planner_endpoints": planner_endpoints, "observer_endpoints": endpoints, "allowed_ids": ids,
            "media_hashes": {s["media_sha256"] for s in left["samples"] + right["samples"]}}


def build(prepared, store):
    from video_os.providers.core import AssetCatalog, PERCEPTION_PROTOCOL
    from .observer import ObserverService
    config, observer = prepared["config"], prepared["config"]["observer"]
    image, asr = config.get("image"), config.get("asr")
    image_kwargs = {}
    if image:
        spec = image["spec"]
        image_kwargs = {"image_base_url": spec["base_url"], "image_model": spec["model"],
                       "image_api_key": credential(image["key"]), "image_retries": spec.get("retries", 0),
                       "image_timeout_seconds": spec.get("timeout_seconds", 120),
                       "image_max_completion_tokens": spec.get("max_completion_tokens", 3072),
                       "count_ocr_enable_thinking": spec.get("enable_thinking")}
    runners = []
    for lane, (endpoint, clients) in enumerate(zip(prepared["observer_endpoints"], prepared["clients"]["lanes"])):
        # The pinned service owns a lock and session registry. Separate services
        # and text clients keep both endpoint lanes independent during inference.
        service = ObserverService(
            catalog=AssetCatalog(prepared["assets"], allowed_roots=[config["media_root"]]),
            output_root=store.root / "perception_sessions" / str(lane), perception_model=observer["model"],
            perception_backend=observer["backend"], asr_backend="whisper" if asr else observer["backend"],
            perception_api_key=credential(observer["key"]), base_url=endpoint,
            provider_timeout_seconds=observer.get("timeout_seconds", 300),
            provider_retries=observer.get("retries", 0), budget=prepared["budget"], protocol=PERCEPTION_PROTOCOL,
            whisper_backend_factory=whisper_factory(asr) if asr else None, **image_kwargs)
        # The shared specialist receipt reads these identities from its service.
        if image:
            service.ocr_perception_model = image["spec"]["model"]
        if asr:
            service.asr_perception_model = asr["spec"]["model"]
        runners.append(EpisodeRunner(service, clients["planner"], extractor=clients.get("extractor"),
            asr_backend="whisper" if asr else observer["backend"], store=store, lane=lane))
    return Calibrator(runners=runners, judges=[Judge(TextRoleClient(c)) for c in prepared["clients"]["judges"]],
        store=store, calibration=prepared["calibration"], validation=prepared["validation"],
        initial=prepared["initial"], search=prepared["search"], validation_policy=prepared["policy"],
        perception_calibration=config.get("perception_calibration", False),
        p_view=prepared["budget"].p_view, allowed_ids=prepared["allowed_ids"],
        resolvers=[ObserverResolver(ProbeRunner(r.service, store), TextRoleClient(c), p_view=prepared["budget"].p_view)
                   for r, c in zip(runners, prepared["clients"]["probe_judges"])])
