"""Credential-free failure categories for bounded infrastructure recovery."""
from __future__ import annotations

import errno
import re
from urllib.error import HTTPError, URLError


RETRYABLE_HTTP = {408, 409, 429, 500, 502, 503, 504}
RETRYABLE_ERRNO = {errno.ECONNRESET, errno.ECONNREFUSED, errno.ECONNABORTED,
                   errno.ETIMEDOUT, errno.EHOSTUNREACH, errno.ENETUNREACH,
                   errno.ENETDOWN, errno.EPIPE}


class ExecutionFailure(RuntimeError):
    def __init__(self, failure: dict):
        self.failure = dict(failure)
        super().__init__("episode execution failed; error artifact saved")


def classify_failure(exc: BaseException) -> dict:
    """Never persist exception messages, request bodies, URLs or credentials."""
    if isinstance(exc, ExecutionFailure):
        return dict(exc.failure)
    chain, seen = [], set()
    current = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    result = {"schema": "moha_failure_v1", "error_type": type(exc).__name__,
              "cause_types": [type(e).__name__ for e in chain],
              "category": "unexpected", "retryable": False}
    # Contract, credential and response-validation errors are deterministic.
    if any(isinstance(e, (ValueError, PermissionError, FileNotFoundError)) for e in chain):
        return {**result, "category": "contract_or_configuration"}
    for error in chain:
        provider_error = any(base.__name__ == "ProviderError" and base.__module__ == "flat.core.errors"
                             for base in type(error).__mro__)
        match = re.match(r"^(?:planner|provider|Whisper) HTTP (\d{3})(?:;|$)", str(error)) if provider_error else None
        status = error.code if isinstance(error, HTTPError) else int(match[1]) if match else None
        if status is not None:
            return {**result, "category": "http", "http_status": status,
                    "retryable": status in RETRYABLE_HTTP}
    for error in chain:
        reason = error.reason if isinstance(error, URLError) else error
        if isinstance(reason, TimeoutError):
            return {**result, "category": "timeout", "retryable": True}
        if isinstance(reason, ConnectionError) or (isinstance(reason, OSError)
                                                   and reason.errno in RETRYABLE_ERRNO):
            return {**result, "category": "connection", "retryable": True}
    return result
