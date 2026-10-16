#!/usr/bin/env python3
"""Lightweight OpenAI-compatible HTTP server for Qwen3.8-Flash-Next MLX OffloadEngine.

Usage:
    python3 server.py --host 127.0.0.1 --port 8090 --budget-pct 20 --alloc-mode even --prefetch on

Example curl requests:
    curl http://127.0.0.1:8090/health
    curl http://127.0.0.1:8090/v1/models
    curl http://127.0.0.1:8090/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":64,"stream":true}'
    curl http://127.0.0.1:8090/v1/responses \\
      -H "Content-Type: application/json" \\
      -d '{"input":"Hello!","max_output_tokens":64,"stream":true}'
"""

from __future__ import annotations

import argparse
import gc
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any, Iterable, Iterator
import uuid

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import loader  # noqa: E402  (sets sys.path for oMLX and mx.set_cache_limit(2GB))
import mlx.core as mx  # noqa: E402

GEN_LOCK = threading.Lock()


def get_stop_token_ids(engine: loader.OffloadEngine) -> set[int]:
    """Collect EOS token IDs from tokenizer, generation_config.json, and <|im_end|>."""
    cached = getattr(engine, "_stop_token_ids", None)
    if cached is not None:
        return set(cached)

    stop_ids: set[int] = set()
    tok = getattr(engine, "tokenizer", None)
    if tok is not None:
        eos = getattr(tok, "eos_token_id", None)
        if isinstance(eos, int):
            stop_ids.add(eos)
        elif isinstance(eos, (list, tuple, set)):
            stop_ids.update(int(x) for x in eos if x is not None)

        vocab = tok.get_vocab() if hasattr(tok, "get_vocab") else {}
        if "<|im_end|>" in vocab:
            stop_ids.add(int(vocab["<|im_end|>"]))
        elif hasattr(tok, "convert_tokens_to_ids"):
            im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
            unk_id = getattr(tok, "unk_token_id", None)
            if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id != unk_id:
                stop_ids.add(im_end_id)

    model_path = getattr(engine, "model_path", None)
    if model_path is not None:
        gen_cfg_path = Path(model_path) / "generation_config.json"
        if gen_cfg_path.exists():
            try:
                gen_cfg = json.loads(gen_cfg_path.read_text(encoding="utf-8"))
                cfg_eos = gen_cfg.get("eos_token_id")
                if isinstance(cfg_eos, int):
                    stop_ids.add(cfg_eos)
                elif isinstance(cfg_eos, list):
                    stop_ids.update(int(x) for x in cfg_eos if isinstance(x, int))
            except Exception:
                pass

    try:
        engine._stop_token_ids = set(stop_ids)
    except Exception:
        pass
    return stop_ids


def _sample_next_token(logits: mx.array, temperature: float, top_p: float) -> mx.array:
    """Sample next token from out.logits (shape [1, L, V]) returning shape [1, 1]."""
    if temperature <= 0.0:
        return mx.argmax(logits[:, -1:, :], axis=-1)

    scaled = logits[:, -1, :].astype(mx.float32) / float(temperature)
    if 0.0 < top_p < 1.0:
        sorted_indices = mx.argsort(-scaled, axis=-1)
        sorted_logits = mx.take_along_axis(scaled, sorted_indices, axis=-1)
        sorted_probs = mx.softmax(sorted_logits, axis=-1)
        cum_probs = mx.cumsum(sorted_probs, axis=-1)
        mask = (cum_probs - sorted_probs) >= float(top_p)
        filtered = mx.where(mask, mx.array(-float("inf"), dtype=sorted_logits.dtype), sorted_logits)
        sampled_pos = mx.random.categorical(filtered, axis=-1)
        return mx.take_along_axis(sorted_indices, sampled_pos[:, None], axis=-1)

    sampled = mx.random.categorical(scaled, axis=-1)
    return sampled[:, None]


def generate_stream(
    engine: loader.OffloadEngine,
    prompt_ids: list[int],
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop_ids: Iterable[int] | None = None,
) -> Iterator[int]:
    """Replicate OffloadEngine.generate_greedy prefill/decode loop as a token ID generator."""
    stop_set = get_stop_token_ids(engine)
    if stop_ids is not None:
        stop_set.update(int(x) for x in stop_ids)

    lm = engine.model.language_model
    cache = lm.make_cache()
    timings: dict[str, Any] = {
        "prefill_s": 0.0,
        "decode_s": 0.0,
        "decode_tok_per_s": 0.0,
        "stopped_by_token": False,
    }
    engine._last_timings = timings
    n_yielded = 0
    t_dec0: float | None = None

    try:
        if max_tokens <= 0:
            return

        # 1. Prefill
        engine.model._ple_cpu_active = False
        inp = mx.array([prompt_ids], dtype=mx.int32)
        t_pre0 = time.perf_counter()
        out = lm(inp, cache=cache)
        next_tok_arr = _sample_next_token(out.logits, temperature, top_p)
        mx.eval(next_tok_arr)
        prefill_sec = time.perf_counter() - t_pre0
        timings["prefill_s"] = round(prefill_sec, 4)
        mx.clear_cache()

        # Initialize CPU history for Layer 1 PLE n-gram hashing during decode
        eos_id = getattr(engine.model, "_ple_eos_id", 248044)
        p_tail = [eos_id, eos_id] + list(prompt_ids)
        engine.model._ple_cpu_history = [p_tail[-2], p_tail[-1], eos_id]
        engine.model._ple_cpu_active = True
        ple_hist = engine.model._ple_cpu_history

        # 2. Decode
        cur_tok = next_tok_arr
        t_dec0 = time.perf_counter()
        for d_idx in range(max_tokens):
            tok_id = int(cur_tok.item())
            if tok_id in stop_set:
                timings["stopped_by_token"] = True
                break

            n_yielded += 1
            yield tok_id

            if d_idx + 1 >= max_tokens:
                break

            ple_hist[2] = tok_id
            out = lm(cur_tok, cache=cache)
            next_tok_arr = _sample_next_token(out.logits, temperature, top_p)
            mx.eval(next_tok_arr)
            cur_tok = next_tok_arr

            ple_hist[0] = ple_hist[1]
            ple_hist[1] = tok_id

            if (d_idx + 1) % 32 == 0:
                mx.clear_cache()
    finally:
        if t_dec0 is not None:
            decode_sec = max(1e-6, time.perf_counter() - t_dec0)
            timings["decode_s"] = round(decode_sec, 4)
            timings["decode_tok_per_s"] = round(n_yielded / decode_sec, 3)
        engine.model._ple_cpu_active = False
        del cache
        gc.collect()
        mx.clear_cache()


def _normalize_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("'messages' must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg:
            raise ValueError("Each message must be an object with a 'role' field")
        content = msg.get("content", "")
        if content is None:
            text_content = ""
        elif isinstance(content, str):
            text_content = content
        elif isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and part.get("type", "text") == "text":
                    parts.append(str(part.get("text", "")))
            text_content = "".join(parts)
        else:
            raise ValueError("Message 'content' must be a string or list of text parts")
        msg_copy = dict(msg)
        msg_copy["content"] = text_content
        normalized.append(msg_copy)
    return normalized


def _normalize_responses_input(
    raw_input: Any,
    instructions: Any = None,
) -> list[dict[str, Any]]:
    """Validate and normalize OpenAI Responses API 'input' and 'instructions' into chat messages."""
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
        return messages

    if isinstance(raw_input, list):
        if not raw_input:
            raise ValueError("'input' list must be non-empty")
        for item in raw_input:
            if not isinstance(item, dict):
                raise ValueError("Each item in 'input' must be an object")
            item_type = item.get("type", "message")
            if item_type is None:
                item_type = "message"
            if item_type != "message":
                raise ValueError(
                    f"Unsupported input item type '{item_type}': only 'message' items are supported"
                )
            role = item.get("role")
            if not isinstance(role, str) or not role:
                raise ValueError("Each message item in 'input' must have a non-empty 'role' field")
            if "content" not in item or item["content"] is None:
                raise ValueError("Each message item in 'input' must have a 'content' field")
            content = item["content"]
            if isinstance(content, str):
                text_content = content
            elif isinstance(content, list):
                parts: list[str] = []
                for part in content:
                    if not isinstance(part, dict):
                        raise ValueError("Each content part in 'input' must be an object")
                    part_type = part.get("type")
                    if part_type not in ("input_text", "output_text", "text"):
                        raise ValueError(
                            f"Unsupported content part type '{part_type}': only text content parts "
                            "('input_text', 'output_text', 'text') are supported"
                        )
                    part_text = part.get("text")
                    if not isinstance(part_text, str):
                        raise ValueError(
                            f"Content part of type '{part_type}' must include a string 'text' field"
                        )
                    parts.append(part_text)
                text_content = "".join(parts)
            else:
                raise ValueError("Message 'content' in 'input' must be a string or list of content parts")
            messages.append({"role": role, "content": text_content})
        return messages

    raise ValueError("'input' must be a string or a list of message items")


def _encode_chat_messages(
    engine: loader.OffloadEngine,
    messages: list[dict[str, Any]],
    req: dict[str, Any],
) -> list[int]:
    """Apply chat template and tokenize messages identically for chat completions and responses."""
    ct_kwargs = req.get("chat_template_kwargs")
    if "enable_thinking" in req and req["enable_thinking"] is not None:
        enable_thinking = bool(req["enable_thinking"])
    elif isinstance(ct_kwargs, dict) and ct_kwargs.get("enable_thinking") is not None:
        enable_thinking = bool(ct_kwargs["enable_thinking"])
    else:
        enable_thinking = False
    formatted = engine.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return engine.tokenizer.encode(formatted, add_special_tokens=False)


def _parse_stop_strings(stop_param: Any) -> list[str]:
    if stop_param is None:
        return []
    if isinstance(stop_param, str):
        return [stop_param] if stop_param else []
    if isinstance(stop_param, list) and all(isinstance(s, str) for s in stop_param):
        return [s for s in stop_param if s]
    raise ValueError("'stop' must be a string or a list of strings")


def _holdback_len(text: str, stop_strings: list[str]) -> int:
    """Compute how many trailing characters of `text` to hold back for stop prefixes or U+FFFD."""
    hold = 0
    for s in stop_strings:
        max_k = min(len(s) - 1, len(text))
        for k in range(max_k, 0, -1):
            if text.endswith(s[:k]):
                if k > hold:
                    hold = k
                break
    if text.endswith("\ufffd"):
        n_rep = len(text) - len(text.rstrip("\ufffd"))
        if n_rep > hold:
            hold = n_rep
    return hold


class OffloadHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        engine: loader.OffloadEngine,
        model_id: str,
    ):
        super().__init__(server_address, OffloadRequestHandler)
        self.engine = engine
        self.model_id = model_id
        self.stop_ids = get_stop_token_ids(engine)
        self.created_ts = int(time.time())


class OffloadRequestHandler(BaseHTTPRequestHandler):
    server: OffloadHTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _send_error_json(self, status: int, message: str, err_type: str = "invalid_request_error") -> None:
        self._send_json(status, {"error": {"message": message, "type": err_type}})

    def _read_json_body(self) -> dict[str, Any]:
        length_str = self.headers.get("Content-Length")
        if not length_str:
            raise ValueError("Missing Content-Length header")
        raw = self.rfile.read(int(length_str))
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Request JSON body must be an object")
        return data

    def _write_sse(self, payload: dict[str, Any] | str) -> None:
        data_str = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        self.wfile.write(f"data: {data_str}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _write_sse_event(self, event_type: str, payload: dict[str, Any]) -> None:
        data_str = json.dumps(payload, ensure_ascii=False)
        self.wfile.write(f"event: {event_type}\ndata: {data_str}\n\n".encode("utf-8"))
        self.wfile.flush()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/health":
            rss_gib = round(loader.get_rss_kb() / (1024**2), 3)
            mlx_active_gib = round(mx.get_active_memory() / (1024**3), 3)
            mlx_peak_gib = round(mx.get_peak_memory() / (1024**3), 3)
            self._send_json(
                200,
                {
                    "status": "ok",
                    "model": self.server.model_id,
                    "busy": GEN_LOCK.locked(),
                    "rss_gib": rss_gib,
                    "mlx_active_gib": mlx_active_gib,
                    "mlx_peak_gib": mlx_peak_gib,
                },
            )
            return

        if path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.model_id,
                            "object": "model",
                            "created": self.server.created_ts,
                            "owned_by": "local",
                        }
                    ],
                },
            )
            return

        self._send_error_json(404, f"Unknown endpoint: {path}", "not_found_error")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path not in ("/v1/chat/completions", "/v1/completions", "/v1/responses"):
            self._send_error_json(404, f"Unknown endpoint: {path}", "not_found_error")
            return

        if path == "/v1/responses":
            self._handle_responses_post()
            return

        try:
            req = self._read_json_body()
            is_chat = path == "/v1/chat/completions"
            engine = self.server.engine

            raw_max = req.get("max_tokens", req.get("max_completion_tokens", 256))
            max_tokens = 256 if raw_max is None else int(raw_max)
            if max_tokens < 0:
                raise ValueError("'max_tokens' must be non-negative")

            temperature = float(req.get("temperature", 0.0) or 0.0)
            top_p = float(1.0 if req.get("top_p") is None else req.get("top_p"))
            stream = bool(req.get("stream", False))
            stop_strings = _parse_stop_strings(req.get("stop"))
            seed = req.get("seed")
            stream_opts = req.get("stream_options") or {}
            include_usage_chunk = bool(
                isinstance(stream_opts, dict) and stream_opts.get("include_usage", False)
            )

            if is_chat:
                messages = _normalize_messages(req.get("messages"))
                prompt_ids = _encode_chat_messages(engine, messages, req)
            else:
                prompt = req.get("prompt")
                if not isinstance(prompt, str) or not prompt:
                    raise ValueError("'prompt' must be a non-empty string")
                prompt_ids = engine.tokenizer.encode(prompt, add_special_tokens=False)

            if not prompt_ids:
                raise ValueError("Encoded prompt produced 0 tokens")
        except Exception as exc:
            self._send_error_json(400, str(exc), "invalid_request_error")
            return

        try:
            self._handle_generation(
                is_chat=is_chat,
                prompt_ids=prompt_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_strings=stop_strings,
                stream=stream,
                seed=seed,
                include_usage_chunk=include_usage_chunk,
            )
        except Exception as exc:
            try:
                self._send_error_json(500, str(exc), "server_error")
            except Exception:
                pass

    def _handle_responses_post(self) -> None:
        try:
            req = self._read_json_body()
            engine = self.server.engine

            if req.get("previous_response_id") is not None:
                raise ValueError(
                    "'previous_response_id' is not supported by this endpoint (server is stateless)"
                )

            tools = req.get("tools")
            if tools is not None:
                if not isinstance(tools, list):
                    raise ValueError("'tools' must be a list")
                if len(tools) > 0:
                    raise ValueError("'tools' is not supported by this endpoint (must be empty or omitted)")

            if "input" not in req or req["input"] is None:
                raise ValueError("Missing required parameter: 'input'")

            instructions = req.get("instructions")
            messages = _normalize_responses_input(req["input"], instructions=instructions)

            raw_max = req.get("max_output_tokens", req.get("max_tokens", 256))
            max_output_tokens = 256 if raw_max is None else int(raw_max)
            if max_output_tokens < 0:
                raise ValueError("'max_output_tokens' must be non-negative")

            temperature = float(req.get("temperature", 0.0) or 0.0)
            top_p = float(1.0 if req.get("top_p") is None else req.get("top_p"))
            stream = bool(req.get("stream", False))
            stop_strings = _parse_stop_strings(req.get("stop"))
            seed = req.get("seed")
            raw_model = req.get("model")
            resp_model = (
                str(raw_model)
                if isinstance(raw_model, str) and raw_model
                else self.server.model_id
            )

            prompt_ids = _encode_chat_messages(engine, messages, req)
            if not prompt_ids:
                raise ValueError("Encoded prompt produced 0 tokens")
        except Exception as exc:
            self._send_error_json(400, str(exc), "invalid_request_error")
            return

        try:
            self._handle_responses_generation(
                prompt_ids=prompt_ids,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_strings=stop_strings,
                stream=stream,
                seed=seed,
                resp_model=resp_model,
                instructions=instructions,
            )
        except Exception as exc:
            try:
                self._send_error_json(500, str(exc), "server_error")
            except Exception:
                pass

    def _handle_responses_generation(
        self,
        *,
        prompt_ids: list[int],
        max_output_tokens: int,
        temperature: float,
        top_p: float,
        stop_strings: list[str],
        stream: bool,
        seed: Any,
        resp_model: str,
        instructions: str | None,
    ) -> None:
        engine = self.server.engine
        resp_id = f"resp_{uuid.uuid4().hex[:24]}"
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        created_at = int(time.time())
        n_prompt = len(prompt_ids)
        seq_num = 0

        def next_seq() -> int:
            nonlocal seq_num
            cur = seq_num
            seq_num += 1
            return cur

        def build_response_obj(
            status: str,
            output_list: list[dict[str, Any]],
            output_text_val: str,
            usage_val: dict[str, Any] | None,
            incomplete_details_val: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            return {
                "id": resp_id,
                "object": "response",
                "created_at": created_at,
                "status": status,
                "error": None,
                "incomplete_details": incomplete_details_val,
                "instructions": instructions,
                "max_output_tokens": max_output_tokens,
                "model": resp_model,
                "output": output_list,
                "output_text": output_text_val,
                "temperature": temperature,
                "top_p": top_p,
                "usage": usage_val,
            }

        with GEN_LOCK:
            if seed is not None:
                mx.random.seed(int(seed))

            gen = generate_stream(
                engine=engine,
                prompt_ids=prompt_ids,
                max_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_ids=self.server.stop_ids,
            )

            generated_ids: list[int] = []
            emitted_len = 0
            final_text = ""
            stopped_by_str = False
            client_disconnected = False

            try:
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True

                    initial_resp = build_response_obj(
                        status="in_progress",
                        output_list=[],
                        output_text_val="",
                        usage_val=None,
                        incomplete_details_val=None,
                    )
                    self._write_sse_event(
                        "response.created",
                        {
                            "type": "response.created",
                            "sequence_number": next_seq(),
                            "response": initial_resp,
                        },
                    )
                    self._write_sse_event(
                        "response.in_progress",
                        {
                            "type": "response.in_progress",
                            "sequence_number": next_seq(),
                            "response": initial_resp,
                        },
                    )
                    self._write_sse_event(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "sequence_number": next_seq(),
                            "output_index": 0,
                            "item": {
                                "type": "message",
                                "id": msg_id,
                                "role": "assistant",
                                "status": "in_progress",
                                "content": [],
                            },
                        },
                    )
                    self._write_sse_event(
                        "response.content_part.added",
                        {
                            "type": "response.content_part.added",
                            "sequence_number": next_seq(),
                            "item_id": msg_id,
                            "output_index": 0,
                            "content_index": 0,
                            "part": {
                                "type": "output_text",
                                "text": "",
                                "annotations": [],
                            },
                        },
                    )

                for tok_id in gen:
                    generated_ids.append(tok_id)
                    full_text = engine.tokenizer.decode(generated_ids)

                    earliest_stop: int | None = None
                    for s in stop_strings:
                        pos = full_text.find(s)
                        if pos != -1 and (earliest_stop is None or pos < earliest_stop):
                            earliest_stop = pos

                    if earliest_stop is not None:
                        final_text = full_text[:earliest_stop]
                        stopped_by_str = True
                        if stream and len(final_text) > emitted_len:
                            piece = final_text[emitted_len:]
                            emitted_len = len(final_text)
                            self._write_sse_event(
                                "response.output_text.delta",
                                {
                                    "type": "response.output_text.delta",
                                    "sequence_number": next_seq(),
                                    "item_id": msg_id,
                                    "output_index": 0,
                                    "content_index": 0,
                                    "delta": piece,
                                },
                            )
                        break

                    final_text = full_text
                    if stream:
                        hold = _holdback_len(full_text, stop_strings)
                        safe_end = len(full_text) - hold
                        if safe_end > emitted_len:
                            piece = full_text[emitted_len:safe_end]
                            emitted_len = safe_end
                            self._write_sse_event(
                                "response.output_text.delta",
                                {
                                    "type": "response.output_text.delta",
                                    "sequence_number": next_seq(),
                                    "item_id": msg_id,
                                    "output_index": 0,
                                    "content_index": 0,
                                    "delta": piece,
                                },
                            )
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                client_disconnected = True
            finally:
                gen.close()

            timings_raw = getattr(engine, "_last_timings", {})
            stopped_by_tok = bool(timings_raw.get("stopped_by_token", False))
            is_stop = (
                stopped_by_str
                or stopped_by_tok
                or len(generated_ids) < max_output_tokens
            )
            finish_reason = "stop" if is_stop else "length"
            resp_status = "completed" if is_stop else "incomplete"
            incomplete_details = (
                None if is_stop else {"reason": "max_output_tokens"}
            )
            n_comp = len(generated_ids)
            usage = {
                "input_tokens": n_prompt,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": n_comp,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": n_prompt + n_comp,
            }
            timings = {
                "prefill_s": timings_raw.get("prefill_s", 0.0),
                "decode_s": timings_raw.get("decode_s", 0.0),
                "decode_tok_per_s": timings_raw.get("decode_tok_per_s", 0.0),
            }

            print(
                f"[server] {self.command} {self.path} | prompt={n_prompt} tok | "
                f"completion={n_comp} tok | prefill={timings['prefill_s']:.3f}s | "
                f"decode={timings['decode_tok_per_s']:.2f} tok/s | finish={finish_reason}"
                + (" (client disconnected)" if client_disconnected else ""),
                flush=True,
            )

            if client_disconnected:
                return

            done_part = {
                "type": "output_text",
                "text": final_text,
                "annotations": [],
            }
            done_item = {
                "type": "message",
                "id": msg_id,
                "role": "assistant",
                "status": "completed",
                "content": [done_part],
            }
            final_resp_obj = build_response_obj(
                status=resp_status,
                output_list=[done_item],
                output_text_val=final_text,
                usage_val=usage,
                incomplete_details_val=incomplete_details,
            )

            if stream:
                try:
                    if not stopped_by_str and len(final_text) > emitted_len:
                        piece = final_text[emitted_len:]
                        emitted_len = len(final_text)
                        self._write_sse_event(
                            "response.output_text.delta",
                            {
                                "type": "response.output_text.delta",
                                "sequence_number": next_seq(),
                                "item_id": msg_id,
                                "output_index": 0,
                                "content_index": 0,
                                "delta": piece,
                            },
                        )

                    self._write_sse_event(
                        "response.output_text.done",
                        {
                            "type": "response.output_text.done",
                            "sequence_number": next_seq(),
                            "item_id": msg_id,
                            "output_index": 0,
                            "content_index": 0,
                            "text": final_text,
                        },
                    )
                    self._write_sse_event(
                        "response.content_part.done",
                        {
                            "type": "response.content_part.done",
                            "sequence_number": next_seq(),
                            "item_id": msg_id,
                            "output_index": 0,
                            "content_index": 0,
                            "part": done_part,
                        },
                    )
                    self._write_sse_event(
                        "response.output_item.done",
                        {
                            "type": "response.output_item.done",
                            "sequence_number": next_seq(),
                            "output_index": 0,
                            "item": done_item,
                        },
                    )
                    self._write_sse_event(
                        "response.completed",
                        {
                            "type": "response.completed",
                            "sequence_number": next_seq(),
                            "response": final_resp_obj,
                        },
                    )
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                return

            self._send_json(200, final_resp_obj)

    def _handle_generation(
        self,
        *,
        is_chat: bool,
        prompt_ids: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_strings: list[str],
        stream: bool,
        seed: Any,
        include_usage_chunk: bool,
    ) -> None:
        engine = self.server.engine
        model_id = self.server.model_id
        cmpl_id = f"{'chatcmpl' if is_chat else 'cmpl'}-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        n_prompt = len(prompt_ids)

        with GEN_LOCK:
            if seed is not None:
                mx.random.seed(int(seed))

            gen = generate_stream(
                engine=engine,
                prompt_ids=prompt_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_ids=self.server.stop_ids,
            )

            generated_ids: list[int] = []
            emitted_len = 0
            final_text = ""
            stopped_by_str = False
            client_disconnected = False

            try:
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True

                    if is_chat:
                        self._write_sse(
                            {
                                "id": cmpl_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_id,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"role": "assistant"},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )

                for tok_id in gen:
                    generated_ids.append(tok_id)
                    full_text = engine.tokenizer.decode(generated_ids)

                    earliest_stop: int | None = None
                    for s in stop_strings:
                        pos = full_text.find(s)
                        if pos != -1 and (earliest_stop is None or pos < earliest_stop):
                            earliest_stop = pos

                    if earliest_stop is not None:
                        final_text = full_text[:earliest_stop]
                        stopped_by_str = True
                        if stream and len(final_text) > emitted_len:
                            piece = final_text[emitted_len:]
                            emitted_len = len(final_text)
                            self._emit_delta_chunk(cmpl_id, created, model_id, is_chat, piece)
                        break

                    final_text = full_text
                    if stream:
                        hold = _holdback_len(full_text, stop_strings)
                        safe_end = len(full_text) - hold
                        if safe_end > emitted_len:
                            piece = full_text[emitted_len:safe_end]
                            emitted_len = safe_end
                            self._emit_delta_chunk(cmpl_id, created, model_id, is_chat, piece)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                client_disconnected = True
            finally:
                gen.close()

            timings_raw = getattr(engine, "_last_timings", {})
            stopped_by_tok = bool(timings_raw.get("stopped_by_token", False))
            finish_reason = (
                "stop"
                if (stopped_by_str or stopped_by_tok or len(generated_ids) < max_tokens)
                else "length"
            )
            n_comp = len(generated_ids)
            usage = {
                "prompt_tokens": n_prompt,
                "completion_tokens": n_comp,
                "total_tokens": n_prompt + n_comp,
            }
            timings = {
                "prefill_s": timings_raw.get("prefill_s", 0.0),
                "decode_s": timings_raw.get("decode_s", 0.0),
                "decode_tok_per_s": timings_raw.get("decode_tok_per_s", 0.0),
            }

            print(
                f"[server] {self.command} {self.path} | prompt={n_prompt} tok | "
                f"completion={n_comp} tok | prefill={timings['prefill_s']:.3f}s | "
                f"decode={timings['decode_tok_per_s']:.2f} tok/s | finish={finish_reason}"
                + (" (client disconnected)" if client_disconnected else ""),
                flush=True,
            )

            if client_disconnected:
                return

            if stream:
                try:
                    if not stopped_by_str and len(final_text) > emitted_len:
                        piece = final_text[emitted_len:]
                        emitted_len = len(final_text)
                        self._emit_delta_chunk(cmpl_id, created, model_id, is_chat, piece)

                    end_choice = (
                        {"index": 0, "delta": {}, "finish_reason": finish_reason}
                        if is_chat
                        else {"index": 0, "text": "", "finish_reason": finish_reason}
                    )
                    self._write_sse(
                        {
                            "id": cmpl_id,
                            "object": "chat.completion.chunk" if is_chat else "text_completion",
                            "created": created,
                            "model": model_id,
                            "choices": [end_choice],
                            "usage": usage,
                            "timings": timings,
                        }
                    )
                    if include_usage_chunk:
                        self._write_sse(
                            {
                                "id": cmpl_id,
                                "object": "chat.completion.chunk" if is_chat else "text_completion",
                                "created": created,
                                "model": model_id,
                                "choices": [],
                                "usage": usage,
                            }
                        )
                    self._write_sse("[DONE]")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                return

            if is_chat:
                resp = {
                    "id": cmpl_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": final_text},
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": usage,
                    "timings": timings,
                }
            else:
                resp = {
                    "id": cmpl_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "text": final_text,
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": usage,
                    "timings": timings,
                }
            self._send_json(200, resp)

    def _emit_delta_chunk(
        self,
        cmpl_id: str,
        created: int,
        model_id: str,
        is_chat: bool,
        piece: str,
    ) -> None:
        if is_chat:
            choice = {"index": 0, "delta": {"content": piece}, "finish_reason": None}
            obj = "chat.completion.chunk"
        else:
            choice = {"index": 0, "text": piece, "finish_reason": None}
            obj = "text_completion"
        self._write_sse(
            {
                "id": cmpl_id,
                "object": obj,
                "created": created,
                "model": model_id,
                "choices": [choice],
            }
        )


def _start_memory_guard(
    start_swap_mb: float,
    max_swap_growth_gb: float,
    max_rss_gb: float,
) -> threading.Thread:
    max_swap_delta_mb = max_swap_growth_gb * 1024.0

    def _guard_loop() -> None:
        while True:
            time.sleep(2.0)
            try:
                rss_gib = loader.get_rss_kb() / (1024**2)
                swap_delta_mb = loader.get_swap_used_mb() - start_swap_mb
                if swap_delta_mb > max_swap_delta_mb:
                    print(
                        f"[server][memory-guard] FATAL: swap growth {swap_delta_mb / 1024.0:.2f} GiB "
                        f"exceeded limit {max_swap_growth_gb:.2f} GiB; exiting (code 3).",
                        file=sys.stderr,
                        flush=True,
                    )
                    os._exit(3)
                if rss_gib > max_rss_gb:
                    print(
                        f"[server][memory-guard] FATAL: RSS {rss_gib:.2f} GiB "
                        f"exceeded limit {max_rss_gb:.2f} GiB; exiting (code 3).",
                        file=sys.stderr,
                        flush=True,
                    )
                    os._exit(3)
            except Exception:
                pass

    t = threading.Thread(target=_guard_loop, name="memory-guard", daemon=True)
    t.start()
    return t


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI-compatible server for Qwen3.8 OffloadEngine")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8090, help="Bind port")
    parser.add_argument("--model-path", type=Path, default=loader.MODEL_PATH, help="Model directory path")
    parser.add_argument("--budget-pct", type=float, default=20.0, help="Per-layer cache budget %% of 512")
    parser.add_argument("--budget-gb", type=float, default=None, help="Total L1..47 cache budget in GB")
    parser.add_argument("--alloc-mode", choices=["even", "proportional"], default="even")
    parser.add_argument("--prefetch", choices=["on", "off"], default="on")
    parser.add_argument("--prefetch-top-m", type=int, default=loader.PREFETCH_TOP_M)
    parser.add_argument("--prefetch-cap", type=int, default=loader.PREFETCH_MAX_CAP)
    parser.add_argument("--model-id", type=str, default="qwen3.8-flash-next-offload")
    parser.add_argument("--max-rss-gb", type=float, default=38.0)
    parser.add_argument("--max-swap-growth-gb", type=float, default=8.0)
    args = parser.parse_args()

    start_swap_mb = loader.get_swap_used_mb()
    _start_memory_guard(
        start_swap_mb=start_swap_mb,
        max_swap_growth_gb=args.max_swap_growth_gb,
        max_rss_gb=args.max_rss_gb,
    )

    budget_pct = None if args.budget_gb is not None else args.budget_pct
    engine = loader.OffloadEngine(
        model_path=Path(args.model_path),
        budget_gb=args.budget_gb,
        budget_pct=budget_pct,
        alloc_mode=args.alloc_mode,
        prefetch=(args.prefetch == "on"),
        prefetch_top_m=args.prefetch_top_m,
        prefetch_cap=args.prefetch_cap,
    )
    rss_gib = loader.get_rss_kb() / (1024**2)
    print(
        f"[server] Loaded model '{args.model_id}' in {engine.load_seconds:.2f}s | "
        f"cache={engine.cache_budget_bytes / (1024**3):.2f} GiB ({engine.total_cached_slots} slots, "
        f"mode={args.alloc_mode}, prefetch={args.prefetch}) | RSS={rss_gib:.2f} GiB",
        flush=True,
    )

    httpd = OffloadHTTPServer((args.host, args.port), engine=engine, model_id=args.model_id)
    print(f"[server] listening on http://{args.host}:{args.port}", flush=True)

    shutting_down = threading.Event()

    def _handle_signal(signum: int, _frame: Any) -> None:
        if not shutting_down.is_set():
            shutting_down.set()
            threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        if hasattr(engine, "close"):
            engine.close()
        print("[server] Shutdown complete.", flush=True)


if __name__ == "__main__":
    main()
