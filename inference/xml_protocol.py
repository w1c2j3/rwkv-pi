"""RWKV XML tool envelope v1. JSON values inside tags are preserved exactly.

No general XML parsing, entity expansion, repair, or arbitrary text extraction.
The canonical form is one <tool_call>{"name":...,"arguments":...}</tool_call>
per call. The plural array envelope and OpenAI function wrapper are accepted on
input because the base model already generates them; the template emits singular.
"""
import json
from jsonschema import Draft202012Validator

MAX_BYTES = 1024 * 1024
MAX_CALLS = 16


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


DECODER = json.JSONDecoder(object_pairs_hook=unique_object, parse_constant=reject_constant)


def call_start(text, *, partial=False):
    """Find a call envelope at a line boundary, outside Markdown code fences.

    Native OpenAI messages can contain prose followed by tool calls. Tags inside
    inline prose or fenced examples remain text. In streams withhold a partial
    opening tag at the beginning of a line until it becomes unambiguous.
    """
    offset, fence = 0, None
    tags = ("<tool_call>", "<tool_calls>")
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip(' \t')
        marker = next((m for m in ('```', '~~~') if stripped.startswith(m)), None)
        if marker:
            fence = None if fence == marker else marker if fence is None else fence
        elif fence is None and (stripped.startswith(tags) or (
                partial and stripped and any(tag.startswith(stripped) for tag in tags))):
            return offset + len(line) - len(stripped)
        offset += len(line)
    return None


def strict_json(text):
    return json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)


def normalize_call(value, tools, choice="auto"):
    if not isinstance(value, dict):
        raise ValueError("Expected a function object")
    if "function" in value:
        if set(value) - {"type", "function", "id"} or value.get("type", "function") != "function":
            raise ValueError("Invalid function wrapper")
        value = value["function"]
    if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
        raise ValueError("Expected name and arguments")
    name, args = value["name"], value["arguments"]
    if not isinstance(name, str) or name not in tools or choice == "none":
        raise ValueError("Tool is not allowed")
    if isinstance(choice, dict) and choice.get("function", {}).get("name") != name:
        raise ValueError("Named tool choice mismatch")
    if isinstance(args, str):
        args = strict_json(args)
    if not isinstance(args, dict):
        raise ValueError("Arguments must be an object")
    Draft202012Validator(tools[name]).validate(args)
    return {"name": name, "arguments": args}


def parse_xml_calls(text, tools, choice="auto", *, partial=False):
    """Return (complete calls, consumed chars). Only a whole XML call sequence.

    In partial mode, an unfinished final frame is withheld. Invalid complete
    frames raise; no unclosed JSON or closing tag is repaired. JSONDecoder finds
    the JSON boundary, so literal closing tags inside strings are safe.
    """
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise ValueError("Tool response exceeds 1 MiB")
    calls, cursor = [], 0
    while cursor < len(text):
        start = cursor
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor == len(text):
            break
        opening = next((tag for tag in ("<tool_call>", "<tool_calls>") if text.startswith(tag, cursor)), None)
        if opening is None:
            if partial and any(tag.startswith(text[cursor:]) for tag in ("<tool_call>", "<tool_calls>")):
                return calls, start
            raise ValueError("Expected only XML tool-call envelopes")
        cursor += len(opening)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        try:
            value, end = DECODER.raw_decode(text, cursor)
        except json.JSONDecodeError:
            if partial:
                return calls, start
            raise ValueError("Incomplete or invalid JSON payload") from None
        closing = "</tool_calls>" if opening == "<tool_calls>" else "</tool_call>"
        cursor = end
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if not text.startswith(closing, cursor):
            if partial and closing.startswith(text[cursor:]):
                return calls, start
            raise ValueError("Missing XML closing tag")
        cursor += len(closing)
        batch = value if opening == "<tool_calls>" else [value]
        if not isinstance(batch, list) or not batch or len(calls) + len(batch) > MAX_CALLS:
            raise ValueError("Expected 1–16 tool calls")
        # Validate an entire plural frame before exposing any of its calls.
        calls.extend([normalize_call(item, tools, choice) for item in batch])
    if not calls and not partial:
        raise ValueError("No XML calls")
    return calls, cursor
