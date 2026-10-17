"""Pure-Python OpenAI-compatible tool-calling utilities for Qwen3.8-Flash-Next.

Handles:
- Tool normalization (Chat Completions nested form and OpenAI Responses flat form)
- tool_choice application ("auto", "none", "required", specific function)
- Qwen3-Coder XML <tool_call> parsing (plus JSON fallback and schema-aware typing)
- Streaming <tool_call> holdback parser (ToolCallStreamParser)
- Chat Completions message normalization & Responses API input item conversion
"""

from __future__ import annotations

import codecs
import json
import re
import sys
from typing import Any
import uuid

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_FUNCTION_OPEN_RE = re.compile(r"<function=\s*([^>\r\n]+?)\s*>")
_PARAM_OPEN_RE = re.compile(r"<parameter=\s*([^>\r\n]+?)\s*>")


def normalize_tools(tools: Any) -> list[dict[str, Any]]:
    """Normalize tool definitions into Chat Completions form.

    Accepts:
      - Chat Completions form: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
      - Responses API flat form: {"type": "function", "name": ..., "description": ..., "parameters": ...}

    Non-function tool types (e.g. "web_search", "code_interpreter") are silently dropped
    and logged to stderr.
    """
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise ValueError("'tools' must be a list")

    normalized: list[dict[str, Any]] = []
    for item in tools:
        if not isinstance(item, dict):
            raise ValueError("Each tool in 'tools' must be an object")

        tool_type = item.get("type", "function")
        if tool_type != "function":
            print(
                f"[tool_calls] Dropping unsupported non-function tool type: {tool_type!r}",
                file=sys.stderr,
                flush=True,
            )
            continue

        if "function" in item:
            fn = item["function"]
            if not isinstance(fn, dict):
                raise ValueError("Tool 'function' field must be an object")
            name = fn.get("name")
            description = fn.get("description", "")
            parameters = fn.get("parameters")
        else:
            name = item.get("name")
            description = item.get("description", "")
            parameters = item.get("parameters")

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Function tool must have a non-empty string 'name'")
        name = name.strip()

        if description is None:
            description = ""
        elif not isinstance(description, str):
            description = str(description)

        if parameters is None:
            parameters = {"type": "object", "properties": {}}
        elif not isinstance(parameters, dict):
            raise ValueError(f"Function '{name}' parameters must be a JSON Schema object")

        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
    return normalized


def _append_system_instruction(
    messages: list[dict[str, Any]],
    instruction: str,
) -> list[dict[str, Any]]:
    """Append a system instruction to messages[0] (or insert a system message at index 0)."""
    msgs = [dict(m) for m in messages]
    if not instruction:
        return msgs
    if msgs and msgs[0].get("role") == "system":
        existing = msgs[0].get("content") or ""
        if not isinstance(existing, str):
            existing = str(existing)
        msgs[0]["content"] = f"{existing}\n\n{instruction}" if existing.strip() else instruction
    else:
        msgs.insert(0, {"role": "system", "content": instruction})
    return msgs


def apply_tool_choice(
    tools: Any,
    tool_choice: Any = "auto",
    messages: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply OpenAI tool_choice to normalized tools and messages.

    Supported tool_choice values:
      - None or "auto": pass all normalized tools, no forced system instruction
      - "none": do not pass tools to template (returns empty tool list)
      - "required": require at least one valid tool; append system instruction that a tool must be called
      - {"type": "function", "function": {"name": X}} or {"type": "function", "name": X}:
        filter tools to X (raising ValueError if X not found) and append system instruction

    Returns:
      (effective_tools, updated_messages)
    """
    normalized_tools = normalize_tools(tools)
    msgs = [dict(m) for m in messages] if messages is not None else []

    if tool_choice is None or tool_choice == "auto":
        return normalized_tools, msgs

    if tool_choice == "none":
        return [], msgs

    if tool_choice == "required":
        if not normalized_tools:
            raise ValueError("tool_choice='required' requires at least one valid function tool")
        instruction = (
            "You MUST call one or more of the provided functions to answer this request. "
            "Respond using a <tool_call> block."
        )
        return normalized_tools, _append_system_instruction(msgs, instruction)

    if isinstance(tool_choice, dict):
        tc_type = tool_choice.get("type")
        if tc_type != "function":
            raise ValueError(f"Unsupported tool_choice type: {tc_type!r}")
        if "function" in tool_choice and isinstance(tool_choice["function"], dict):
            target_name = tool_choice["function"].get("name")
        else:
            target_name = tool_choice.get("name")
        if not isinstance(target_name, str) or not target_name.strip():
            raise ValueError("tool_choice function 'name' must be a non-empty string")
        target_name = target_name.strip()

        filtered = [t for t in normalized_tools if t["function"]["name"] == target_name]
        if not filtered:
            raise ValueError(
                f"Function '{target_name}' specified in tool_choice was not found in 'tools'"
            )
        instruction = (
            f"You MUST call the function '{target_name}' to answer this request. "
            f"Respond using a <tool_call> block with <function={target_name}>."
        )
        return filtered, _append_system_instruction(msgs, instruction)

    raise ValueError(f"Invalid tool_choice value: {tool_choice!r}")


def _build_param_schema_map(tools: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Build {function_name: {param_name: param_schema_dict}} lookup map from tools."""
    schema_map: dict[str, dict[str, Any]] = {}
    if not tools or not isinstance(tools, list):
        return schema_map

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name")
        params = fn.get("parameters")
        if isinstance(name, str) and name.strip() and isinstance(params, dict):
            props = params.get("properties")
            if isinstance(props, dict):
                schema_map[name.strip()] = props
    return schema_map


def _get_param_schema_type(
    schema_map: dict[str, dict[str, Any]],
    fn_name: str,
    param_name: str,
) -> str | None:
    """Return the JSON schema type for a function parameter, or None if unknown."""
    fn_props = schema_map.get(fn_name)
    if not isinstance(fn_props, dict):
        return None
    prop_schema = fn_props.get(param_name)
    if not isinstance(prop_schema, dict):
        return None

    raw_type = prop_schema.get("type")
    if isinstance(raw_type, str):
        return raw_type
    if isinstance(raw_type, list):
        non_null = [t for t in raw_type if isinstance(t, str) and t != "null"]
        if non_null:
            return non_null[0]
        str_types = [t for t in raw_type if isinstance(t, str)]
        return str_types[0] if str_types else None

    for union_key in ("anyOf", "oneOf"):
        variants = prop_schema.get(union_key)
        if isinstance(variants, list):
            for v in variants:
                if isinstance(v, dict) and isinstance(v.get("type"), str) and v["type"] != "null":
                    return v["type"]
    return None


def _strip_single_outer_newline(raw_val: str) -> str:
    """Strip at most one leading newline and at most one trailing newline."""
    val = raw_val
    if val.startswith("\r\n"):
        val = val[2:]
    elif val.startswith("\n"):
        val = val[1:]

    if val.endswith("\r\n"):
        val = val[:-2]
    elif val.endswith("\n"):
        val = val[:-1]
    return val


def _convert_param_value(
    raw_val: str,
    fn_name: str,
    param_name: str,
    schema_map: dict[str, dict[str, Any]],
) -> Any:
    """Convert raw parameter string according to the tool's JSON schema type."""
    val = _strip_single_outer_newline(raw_val)
    param_type = _get_param_schema_type(schema_map, fn_name, param_name)

    if param_type == "string":
        return val

    if param_type in ("integer", "number", "boolean", "object", "array", "null"):
        try:
            return json.loads(val)
        except Exception:
            return val

    # Unknown parameter or no schema provided: try json.loads, else fall back to raw string
    try:
        return json.loads(val)
    except Exception:
        return val


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return _THINK_BLOCK_RE.sub("", text)


def _parse_single_tool_call_block(
    block: str,
    schema_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Parse one <tool_call>...</tool_call> body into one or more tool call objects."""
    results: list[dict[str, Any]] = []
    fn_matches = list(_FUNCTION_OPEN_RE.finditer(block))

    if fn_matches:
        for i, fn_match in enumerate(fn_matches):
            fn_name = fn_match.group(1).strip().strip("'\"")
            if not fn_name:
                continue
            fn_start = fn_match.end()
            fn_end = fn_matches[i + 1].start() if i + 1 < len(fn_matches) else len(block)
            fn_body = block[fn_start:fn_end]
            if "</function>" in fn_body:
                fn_body = fn_body.split("</function>", 1)[0]

            param_matches = list(_PARAM_OPEN_RE.finditer(fn_body))
            args_dict: dict[str, Any] = {}
            for j, pm in enumerate(param_matches):
                p_name = pm.group(1).strip().strip("'\"")
                if not p_name:
                    continue
                p_start = pm.end()
                p_end = param_matches[j + 1].start() if j + 1 < len(param_matches) else len(fn_body)
                p_chunk = fn_body[p_start:p_end]
                if "</parameter>" in p_chunk:
                    raw_val = p_chunk.split("</parameter>", 1)[0]
                else:
                    raw_val = p_chunk
                args_dict[p_name] = _convert_param_value(raw_val, fn_name, p_name, schema_map)

            results.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": json.dumps(args_dict, ensure_ascii=False),
                    },
                }
            )
        return results

    # Fallback: JSON-style <tool_call>{"name": ..., "arguments": {...}}</tool_call>
    candidate = block.strip()
    if not candidate:
        return results

    try:
        obj = json.loads(candidate)
    except Exception:
        return results

    items = obj if isinstance(obj, list) else [obj]
    for entry in items:
        if not isinstance(entry, dict):
            continue
        if isinstance(entry.get("function"), dict):
            entry = entry["function"]
        fn_name = entry.get("name")
        if not isinstance(fn_name, str) or not fn_name.strip():
            continue
        fn_name = fn_name.strip()
        raw_args = entry.get("arguments", entry.get("parameters", {}))
        if isinstance(raw_args, dict):
            args_json = json.dumps(raw_args, ensure_ascii=False)
        elif isinstance(raw_args, str):
            try:
                parsed_args = json.loads(raw_args)
                if isinstance(parsed_args, dict):
                    args_json = json.dumps(parsed_args, ensure_ascii=False)
                else:
                    args_json = raw_args
            except Exception:
                args_json = raw_args
        else:
            args_json = "{}"

        results.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": fn_name,
                    "arguments": args_json,
                },
            }
        )
    return results


def parse_tool_calls(
    text: str,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Parse Qwen3-Coder XML (or JSON fallback) tool calls from model output.

    Returns:
      (content_text, tool_calls)
      where content_text is the text before the first <tool_call>, stripped
      (or None if empty when tool_calls exist), and tool_calls is a list of:
      {"id": "call_<hex>", "type": "function", "function": {"name": ..., "arguments": JSON str}}
    """
    if not text:
        return "", []

    cleaned = _strip_think_blocks(text)
    first_tc = cleaned.find(TOOL_CALL_OPEN)
    if first_tc == -1:
        return cleaned.strip(), []

    prefix = cleaned[:first_tc].strip()
    rest = cleaned[first_tc:]
    schema_map = _build_param_schema_map(tools)

    calls: list[dict[str, Any]] = []
    segments = rest.split(TOOL_CALL_OPEN)[1:]
    for seg in segments:
        if TOOL_CALL_CLOSE in seg:
            block = seg.split(TOOL_CALL_CLOSE, 1)[0]
        else:
            block = seg
        calls.extend(_parse_single_tool_call_block(block, schema_map))

    if calls:
        content_text: str | None = prefix if prefix else None
        return content_text, calls

    return cleaned.strip(), []


def _tag_prefix_holdback(text: str, tags: tuple[str, ...] = (TOOL_CALL_OPEN, THINK_OPEN)) -> int:
    """Return number of trailing characters in `text` that match a prefix of any tag in `tags`."""
    hold = 0
    for tag in tags:
        max_k = min(len(tag) - 1, len(text))
        for k in range(max_k, 0, -1):
            if text.endswith(tag[:k]):
                if k > hold:
                    hold = k
                break
    return hold


class ToolCallStreamParser:
    """Incremental stream parser that holds back potential <tool_call> prefixes and buffers tool calls.

    Usage:
      parser = ToolCallStreamParser(tools)
      for delta in chunks:
          for kind, piece in parser.feed(delta):
              # kind == "text"
              emit(piece)
      remaining_text, tool_calls = parser.finish()
    """

    def __init__(self, tools: list[dict[str, Any]] | None = None) -> None:
        self.tools = tools
        self._utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._raw_text = ""
        self._work_buf = ""
        self._emitted_len = 0
        self._in_tool_call = False

    def feed(self, delta_text: str | bytes) -> list[tuple[str, str]]:
        """Feed a delta chunk and return a list of ('text', safe_str) events."""
        if isinstance(delta_text, bytes):
            delta_str = self._utf8_decoder.decode(delta_text, final=False)
        elif isinstance(delta_text, str):
            delta_str = delta_text
        else:
            raise TypeError("delta_text must be str or bytes")

        if not delta_str:
            return []

        self._raw_text += delta_str
        if self._in_tool_call:
            return []

        self._work_buf += delta_str

        # Strip any completed <think>...</think> blocks
        while True:
            m = _THINK_BLOCK_RE.search(self._work_buf)
            if not m:
                break
            self._work_buf = self._work_buf[: m.start()] + self._work_buf[m.end() :]

        # If an unclosed <think> is present, only consider text before it for now
        think_pos = self._work_buf.find(THINK_OPEN)
        if think_pos != -1:
            active = self._work_buf[:think_pos]
        else:
            active = self._work_buf

        # Check if <tool_call> has appeared in active text
        tc_pos = active.find(TOOL_CALL_OPEN)
        if tc_pos != -1:
            self._in_tool_call = True
            prefix = active[:tc_pos].strip()
            if len(prefix) > self._emitted_len:
                piece = prefix[self._emitted_len :]
                self._emitted_len = len(prefix)
                return [("text", piece)]
            return []

        # If no <think> is open, hold back any trailing prefix of <tool_call> or <think>
        if think_pos == -1:
            hold = _tag_prefix_holdback(active, (TOOL_CALL_OPEN, THINK_OPEN))
            candidate = active[: len(active) - hold] if hold > 0 else active
        else:
            candidate = active

        # Strip leading whitespace (before first emission) and hold back trailing whitespace
        # so that if <tool_call> immediately follows, the emitted content equals prefix.strip().
        safe = candidate.strip()
        if len(safe) > self._emitted_len:
            piece = safe[self._emitted_len :]
            self._emitted_len = len(safe)
            return [("text", piece)]
        return []

    def finish(self) -> tuple[str, list[dict[str, Any]]]:
        """Finalize stream parsing and return (remaining_text, tool_calls)."""
        tail = self._utf8_decoder.decode(b"", final=True)
        if tail:
            self._raw_text += tail

        content_text, tool_calls = parse_tool_calls(self._raw_text, self.tools)
        target_text = content_text or ""
        if len(target_text) > self._emitted_len:
            remaining_text = target_text[self._emitted_len :]
            self._emitted_len = len(target_text)
        else:
            remaining_text = ""
        return remaining_text, tool_calls


def _extract_text_content(content: Any, *, allow_part_types: tuple[str, ...] | None = None) -> str:
    """Convert message content (str, list of text parts, or None) to a string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                if allow_part_types is not None:
                    raise ValueError("Each content part in 'input' must be an object")
                parts.append(part)
            elif isinstance(part, dict):
                part_type = part.get("type", "text" if allow_part_types is None else None)
                if allow_part_types is not None and part_type not in allow_part_types:
                    raise ValueError(
                        f"Unsupported content part type '{part_type}': only text content parts "
                        f"{allow_part_types} are supported"
                    )
                if allow_part_types is None and part_type not in ("text", "input_text", "output_text"):
                    continue
                part_text = part.get("text") if allow_part_types is not None else part.get("text", "")
                if allow_part_types is not None and not isinstance(part_text, str):
                    raise ValueError(
                        f"Content part of type '{part_type}' must include a string 'text' field"
                    )
                parts.append(str(part_text))
            else:
                if allow_part_types is not None:
                    raise ValueError("Each content part in 'input' must be an object")
        return "".join(parts)
    raise ValueError("Message 'content' must be a string or list of text parts")


def _normalize_single_tool_call_for_template(tc: Any) -> dict[str, Any]:
    """Normalize an assistant tool_call entry so function.arguments is a dict mapping for Jinja."""
    if not isinstance(tc, dict):
        raise ValueError("Each entry in assistant 'tool_calls' must be an object")

    tc_id = tc.get("id") or tc.get("call_id") or f"call_{uuid.uuid4().hex[:24]}"
    if isinstance(tc.get("function"), dict):
        fn = tc["function"]
        fn_name = fn.get("name")
        raw_args = fn.get("arguments", {})
    else:
        fn_name = tc.get("name")
        raw_args = tc.get("arguments", {})

    if not isinstance(fn_name, str) or not fn_name.strip():
        raise ValueError("Assistant tool_call must include a non-empty function 'name'")
    fn_name = fn_name.strip()

    if isinstance(raw_args, dict):
        args_dict: dict[str, Any] = dict(raw_args)
    elif isinstance(raw_args, str):
        if raw_args == "":
            args_dict = {}
        else:
            try:
                parsed = json.loads(raw_args)
                if isinstance(parsed, dict):
                    args_dict = parsed
                else:
                    args_dict = {"raw": raw_args}
            except Exception:
                args_dict = {"raw": raw_args}
    elif raw_args is None:
        args_dict = {}
    else:
        args_dict = {"raw": str(raw_args)}

    return {
        "id": str(tc_id),
        "type": "function",
        "function": {
            "name": fn_name,
            "arguments": args_dict,
        },
    }


def _consolidate_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Consolidate any system messages into a single system message at index 0."""
    sys_contents: list[str] = []
    other_msgs: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "system":
            c = m.get("content", "")
            if isinstance(c, str) and c.strip():
                sys_contents.append(c.strip())
            elif isinstance(c, str) and c:
                sys_contents.append(c)
        else:
            other_msgs.append(m)

    if sys_contents:
        return [{"role": "system", "content": "\n\n".join(sys_contents)}] + other_msgs
    # Preserve a single empty system message at index 0 if the input only had an empty system msg at 0
    if messages and messages[0].get("role") == "system":
        return [{"role": "system", "content": ""}] + other_msgs
    return other_msgs


def convert_chat_messages(messages: Any) -> list[dict[str, Any]]:
    """Validate and convert incoming Chat Completions messages for chat_template.jinja.

    - Maps role 'developer' -> 'system' and consolidates system messages at index 0
    - Converts content None -> '', list of text parts -> joined str
    - Converts assistant tool_calls arguments from JSON string -> dict (or {'raw': str} on error)
    - Preserves role='tool' messages as {'role': 'tool', 'content': str}
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("'messages' must be a non-empty list")

    normalized: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg:
            raise ValueError("Each message must be an object with a 'role' field")
        role = msg.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError("Each message must have a non-empty string 'role' field")
        if role == "developer":
            role = "system"

        content = msg.get("content", "")
        text_content = _extract_text_content(content)

        if role == "tool":
            tool_msg: dict[str, Any] = {"role": "tool", "content": text_content}
            if "tool_call_id" in msg:
                tool_msg["tool_call_id"] = msg["tool_call_id"]
            if "name" in msg:
                tool_msg["name"] = msg["name"]
            normalized.append(tool_msg)
            continue

        msg_copy = dict(msg)
        msg_copy["role"] = role
        msg_copy["content"] = text_content

        if role == "assistant" and "tool_calls" in msg_copy:
            raw_tcs = msg_copy["tool_calls"]
            if raw_tcs is None:
                del msg_copy["tool_calls"]
            elif isinstance(raw_tcs, list):
                msg_copy["tool_calls"] = [
                    _normalize_single_tool_call_for_template(tc) for tc in raw_tcs
                ]
            else:
                raise ValueError("Assistant 'tool_calls' must be a list")

        normalized.append(msg_copy)

    return _consolidate_system_messages(normalized)


normalize_chat_messages = convert_chat_messages


def _extract_tool_output_str(raw_output: Any) -> str:
    """Convert Responses API function_call_output 'output' to a string."""
    if isinstance(raw_output, str):
        return raw_output
    if isinstance(raw_output, list):
        parts: list[str] = []
        for part in raw_output:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part.get("output"), str):
                    parts.append(part["output"])
                else:
                    parts.append(json.dumps(part, ensure_ascii=False))
            else:
                raise ValueError("Each part in function_call_output 'output' list must be str or object")
        return "".join(parts)
    if isinstance(raw_output, (dict, int, float, bool)):
        return json.dumps(raw_output, ensure_ascii=False)
    raise ValueError("function_call_output 'output' must be a string or list of text parts")


def convert_responses_input(
    raw_input: Any,
    instructions: Any = None,
) -> list[dict[str, Any]]:
    """Validate and convert OpenAI Responses API 'input' and 'instructions' into chat messages.

    Supports:
      - raw_input as non-empty string
      - raw_input as non-empty list of items:
        * {"type": "message", "role": "user"|"assistant"|"system"|"developer", "content": ...}
        * {"type": "function_call", "call_id": ..., "name": ..., "arguments": ...}
          (consecutive function_call items are merged into a single assistant message, and merged
          into an immediately preceding assistant message item if present)
        * {"type": "function_call_output", "call_id": ..., "output": ...} -> {"role": "tool", "content": str}
        * {"type": "reasoning", ...} -> skipped
    """
    if raw_input is None:
        raise ValueError("Missing required parameter: 'input'")

    messages: list[dict[str, Any]] = []
    if instructions is not None:
        if not isinstance(instructions, str):
            raise ValueError("'instructions' must be a string")
        if instructions:
            messages.append({"role": "system", "content": instructions})

    if isinstance(raw_input, str):
        if not raw_input:
            raise ValueError("'input' string must be non-empty")
        messages.append({"role": "user", "content": raw_input})
        return _consolidate_system_messages(messages)

    if isinstance(raw_input, list):
        if not raw_input:
            raise ValueError("'input' list must be non-empty")

        prev_was_assistant = False
        for item in raw_input:
            if not isinstance(item, dict):
                raise ValueError("Each item in 'input' must be an object")

            item_type = item.get("type", "message")
            if item_type is None:
                item_type = "message"

            if item_type == "reasoning":
                continue

            if item_type == "message":
                role = item.get("role")
                if not isinstance(role, str) or not role:
                    raise ValueError("Each message item in 'input' must have a non-empty 'role' field")
                if role == "developer":
                    role = "system"
                if role not in ("user", "assistant", "system", "tool"):
                    raise ValueError(f"Unsupported message role '{role}' in 'input'")

                if "content" not in item or item["content"] is None:
                    if role == "assistant" and item.get("tool_calls"):
                        text_content = ""
                    else:
                        raise ValueError("Each message item in 'input' must have a 'content' field")
                else:
                    content = item["content"]
                    if isinstance(content, str):
                        text_content = content
                    elif isinstance(content, list):
                        text_content = _extract_text_content(
                            content,
                            allow_part_types=("input_text", "output_text", "text"),
                        )
                    else:
                        raise ValueError(
                            "Message 'content' in 'input' must be a string or list of content parts"
                        )

                msg_entry: dict[str, Any] = {"role": role, "content": text_content}
                if role == "assistant" and item.get("tool_calls"):
                    if not isinstance(item["tool_calls"], list):
                        raise ValueError("Assistant 'tool_calls' must be a list")
                    msg_entry["tool_calls"] = [
                        _normalize_single_tool_call_for_template(tc) for tc in item["tool_calls"]
                    ]
                messages.append(msg_entry)
                prev_was_assistant = role == "assistant"
                continue

            if item_type == "function_call":
                tc_entry = _normalize_single_tool_call_for_template(item)
                if prev_was_assistant and messages and messages[-1].get("role") == "assistant":
                    messages[-1].setdefault("tool_calls", []).append(tc_entry)
                else:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [tc_entry],
                        }
                    )
                prev_was_assistant = True
                continue

            if item_type == "function_call_output":
                if "output" not in item or item["output"] is None:
                    raise ValueError("function_call_output item must have an 'output' field")
                out_str = _extract_tool_output_str(item["output"])
                tool_msg: dict[str, Any] = {"role": "tool", "content": out_str}
                if "call_id" in item and item["call_id"] is not None:
                    tool_msg["tool_call_id"] = str(item["call_id"])
                messages.append(tool_msg)
                prev_was_assistant = False
                continue

            raise ValueError(
                f"Unsupported input item type '{item_type}': supported types are "
                "'message', 'function_call', 'function_call_output', 'reasoning'"
            )

        return _consolidate_system_messages(messages)

    raise ValueError("'input' must be a string or a list of message items")


normalize_responses_input = convert_responses_input
