"""Recover an explicit terminal answer without making another model call."""
from __future__ import annotations

import json
import re
from collections.abc import Collection


ANSWER_PARSING_POLICY = "moha_terminal_json_v2"


def final_answer_response_format(option_labels: Collection[str]) -> dict:
    """Constrain a no-tool terminal call to the two accepted answer shapes."""
    labels = list(option_labels)
    if not labels or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("option labels must be non-empty strings")
    if len(set(labels)) != len(labels):
        raise ValueError("option labels must be unique")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "moha_terminal_answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["answered", "abstained"]},
                    "answer": {"anyOf": [{"type": "string", "enum": labels}, {"type": "null"}]},
                },
                "required": ["status", "answer"],
                "additionalProperties": False,
            },
        },
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def terminal_json_answer(content: object, option_labels: Collection[str]) -> dict | None:
    """Read a complete answer object at the end of the assistant's text.

    Explanatory prose or a JSON code fence may precede the object. Never infer
    an option from the explanation, accept an object inside tool/reasoning
    markup, or repair malformed JSON. Existing structured and text-pattern
    handling remains the caller's responsibility.
    """
    if not isinstance(content, str):
        return None
    text = content.strip()
    if text.endswith("```"):
        opening = text.rfind("```", 0, len(text) - 3)
        if opening < 0:
            return None
        language, separator, body = text[opening + 3:-3].partition("\n")
        if not separator or language.strip().lower() not in ("", "json"):
            return None
        text = text[:opening] + body.rstrip()
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    for match in re.finditer(r"\{", text):
        start = match.start()
        prefix = text[:start].lower()
        if any(prefix.rfind("<" + tag + ">") > prefix.rfind("</" + tag + ">")
               for tag in ("tool_call", "think")):
            continue
        try:
            value, end = decoder.raw_decode(text, start)
        except (ValueError, json.JSONDecodeError):
            continue
        if text[end:].strip() or not isinstance(value, dict):
            continue
        status, answer = value.get("status"), value.get("answer")
        if status == "answered" and isinstance(answer, str) and answer in option_labels:
            return {"status": "answered", "answer": answer}
        if status == "abstained" and "answer" in value and answer is None:
            return {"status": "abstained", "answer": None}
    return None


def terminal_tool_call_answer(name: object, arguments: object,
                              option_labels: Collection[str]) -> dict | None:
    """Recover one explicit answer emitted through an unadvertised terminal tool.

    Some OpenAI-compatible models serialize a requested final answer as an
    ``answer`` or ``final_answer`` tool call even when that tool was never
    advertised. This recognizes only the exact terminal payload and never
    executes the hallucinated tool.
    """
    if name not in {"answer", "final_answer"}:
        return None
    if isinstance(arguments, str):
        try:
            value = json.loads(arguments, object_pairs_hook=_unique_object)
        except (ValueError, json.JSONDecodeError):
            return None
    elif isinstance(arguments, dict):
        value = arguments
    else:
        return None
    if set(value) != {"status", "answer"}:
        return None
    status, answer = value["status"], value["answer"]
    if status == "answered" and isinstance(answer, str) and answer in option_labels:
        return {"status": "answered", "answer": answer}
    # Qwen's XML-to-tool normalization represents an empty null parameter as
    # an empty string. The explicit abstained status keeps this unambiguous.
    if status == "abstained" and answer in (None, ""):
        return {"status": "abstained", "answer": None}
    return None
