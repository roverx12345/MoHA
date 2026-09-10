"""Tool novelty control, independent of verification and evidence storage."""
from __future__ import annotations
import copy
import json
from .models import digest

NO_NOVELTY_POLICY = "moha_visible_memory_novelty_v1"


def visible_memory_payloads(messages):
    values = []
    for message in messages:
        if message.get("role") != "tool" or not isinstance(message.get("content"), str):
            continue
        try:
            value = json.loads(message["content"])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and not value.get("isError"):
            payload = {k: value[k] for k in ("result_memory", "working_memory") if k in value}
            if payload:
                values.append(payload)
    return values


class NoNoveltyController:
    """Suppress unchanged visible results; never confuse eviction with redundancy."""
    def __init__(self):
        self.versions = None
        self.seen = set()
        self.duplicate_streak = 0
        self.masked_payload = None

    @staticmethod
    def visible(payload, visible_payloads):
        return any(all(k in v and v[k] == value for k, value in payload.items()) for v in visible_payloads)

    def refresh(self, versions, visible_payloads):
        if self.versions != versions:
            self.versions = copy.deepcopy(versions)
            self.duplicate_streak = 0
            self.masked_payload = None
        elif self.masked_payload is not None and not self.visible(self.masked_payload, visible_payloads):
            # Restoring an evicted result is a legitimate use of memory.
            self.masked_payload = None
            self.duplicate_streak = 0

    @property
    def masked(self):
        return self.masked_payload is not None

    def other_action(self):
        self.duplicate_streak = 0

    def read(self, payload, versions, visible_payloads):
        self.refresh(versions, visible_payloads)
        fingerprint = digest(payload)
        redundant = fingerprint in self.seen and self.visible(payload, visible_payloads)
        if self.masked or redundant:
            if redundant:
                self.duplicate_streak += 1
                if self.duplicate_streak >= 2:
                    self.masked_payload = copy.deepcopy(payload)
            return {"no_novelty": redundant, "memory_read_masked": self.masked,
                    "read_fingerprint": fingerprint, "ledger_versions": copy.deepcopy(versions),
                    "message": "No new ledger content is being returned. Previously read evidence remains in context. "
                               "Do not repeat this read; continue with another useful action or submit your answer."}
        restored = fingerprint in self.seen
        self.seen.add(fingerprint)
        self.duplicate_streak = 0
        return {**copy.deepcopy(payload), "no_novelty": False, "restored_after_eviction": restored,
                "read_fingerprint": fingerprint, "ledger_versions": copy.deepcopy(versions)}

    def inventory(self):
        return {"policy": NO_NOVELTY_POLICY, "memory_read_available": not self.masked,
                "consecutive_redundant_reads": self.duplicate_streak}
