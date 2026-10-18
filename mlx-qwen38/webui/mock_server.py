#!/usr/bin/env python3
"""Stdlib-only mock server for testing the local Web UI on 127.0.0.1:8097."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HOST = "127.0.0.1"
PORT = 8097
WEBUI_DIR = Path(__file__).resolve().parent

WEBUI_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".json": "application/json",
}

STATE_LOCK = threading.Lock()
BUSY = False
REQ_COUNT = 12
DEMAND_MISSES = 98
LRU_HITS = 960
PREFETCH_HITS = 142
DEMAND_READ_BYTES = 411_041_792
PREFETCH_READ_BYTES = 595_591_168

LAST_REQUEST: dict[str, Any] = {
    "endpoint": "/v1/chat/completions",
    "finished_at": int(time.time()) - 15,
    "prompt_tokens": 64,
    "completion_tokens": 92,
    "finish_reason": "stop",
    "prefill_s": 0.32,
    "prefill_tok_per_s": 200.0,
    "decode_s": 3.58,
    "decode_tok_per_s": 25.7,
    "request_s": 3.90,
}


def _set_busy(val: bool) -> None:
    global BUSY
    with STATE_LOCK:
        BUSY = val


def _get_busy() -> bool:
    with STATE_LOCK:
        return BUSY


class MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(raw)

    def _send_error_json(self, status: int, message: str, err_type: str = "invalid_request_error") -> None:
        self._send_json(status, {"error": {"message": message, "type": err_type}})

    def _send_static(self, rel: str) -> None:
        base = WEBUI_DIR.resolve()
        target = (base / rel).resolve()
        if base not in target.parents or not target.is_file() or target.suffix not in WEBUI_TYPES:
            self._send_error_json(404, f"Unknown endpoint: {self.path}", "not_found_error")
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", WEBUI_TYPES[target.suffix])
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/ui", "/ui/"):
            self._send_static("index.html")
            return
        if path.startswith("/ui/"):
            self._send_static(path[len("/ui/") :])
            return
        if path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": "qwen3.5-397b-a17b-mlx-4bit", "object": "model", "owned_by": "local"},
                        {"id": "qwen3-30b-a3b-mlx-4bit", "object": "model", "owned_by": "local"},
                    ],
                },
            )
            return
        if path in ("/health", "/healthz"):
            self._send_json(
                200,
                {
                    "status": "ok",
                    "busy": _get_busy(),
                    "rss_gib": 14.382,
                    "mlx_active_gib": 11.904,
                    "mlx_peak_gib": 12.640,
                },
            )
            return
        if path == "/api/status":
            with STATE_LOCK:
                total_decode = LRU_HITS + PREFETCH_HITS + DEMAND_MISSES
                hit_rate = round((LRU_HITS + PREFETCH_HITS) / total_decode, 4) if total_decode else None
                cache = {
                    "hit_rate": hit_rate,
                    "decode_requests": total_decode,
                    "lru_hits": LRU_HITS,
                    "prefetch_hits": PREFETCH_HITS,
                    "demand_misses": DEMAND_MISSES,
                    "demand_read_bytes": DEMAND_READ_BYTES,
                    "prefetch_read_bytes": PREFETCH_READ_BYTES,
                }
                last_req = dict(LAST_REQUEST)
                is_busy = BUSY
            self._send_json(
                200,
                {
                    "status": "ok",
                    "busy": is_busy,
                    "model": "qwen3.5-397b-a17b-mlx-4bit",
                    "memory": {
                        "rss_gib": 14.382,
                        "mlx_active_gib": 11.904,
                        "mlx_peak_gib": 12.640,
                        "swap_used_mb": 128.5,
                    },
                    "expert_cache_since_start": cache,
                    "last_request": last_req,
                    "context_length": 32768,
                },
            )
            return
        self._send_error_json(404, f"Unknown endpoint: {self.path}", "not_found_error")

    def do_POST(self) -> None:  # noqa: N802
        global REQ_COUNT, DEMAND_MISSES, LRU_HITS, PREFETCH_HITS, DEMAND_READ_BYTES, PREFETCH_READ_BYTES
        path = self.path.split("?", 1)[0]
        if path != "/v1/chat/completions":
            self._send_error_json(404, f"Unknown endpoint: {self.path}", "not_found_error")
            return

        t_req0 = time.perf_counter()
        content_len = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_len) if content_len > 0 else b""
        try:
            req = json.loads(raw_body.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            self._send_error_json(400, f"Invalid JSON body: {exc}")
            return

        messages = req.get("messages") or []
        if not isinstance(messages, list) or not messages:
            self._send_error_json(400, "messages must be a non-empty list")
            return

        last_user = ""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                last_user = str(m.get("content") or "")
                break

        if "[ERROR]" in last_user:
            self._send_error_json(500, "模擬伺服器內部錯誤 (Simulated model execution failure)", "server_error")
            return

        model_id = str(req.get("model") or "qwen3.5-397b-a17b-mlx-4bit")
        stream = bool(req.get("stream", True))
        enable_thinking = bool(req.get("enable_thinking", False))
        if not enable_thinking and isinstance(req.get("chat_template_kwargs"), dict):
            enable_thinking = bool(req["chat_template_kwargs"].get("enable_thinking", False))

        want_tool = "[TOOL]" in last_user
        stop_words = req.get("stop") or []
        if isinstance(stop_words, str):
            stop_words = [stop_words]

        reasoning_pieces = [
            "正在分析使用者的問題：",
            f"「{last_user[:40]}」...\n",
            "1. 檢查 MoE expert cache 狀態與記憶體配置。\n",
            "2. 準備 Markdown 格式回覆與程式碼範例，並確保安全性。",
        ]
        answer_pieces = [
            "這是來自 **Mock Server** 的測試回覆（支援 *斜體* 與 `inline code`）。\n\n",
            "```python\ndef hello_moe(x: int) -> int:\n    return x * 2\n```\n\n",
            "詳情可參考 [OpenAI API 文件](https://platform.openai.com/docs) 或測試 HTML 跳脫：`<script>alert('xss')</script>`。",
        ]
        if stop_words:
            joined = "".join(answer_pieces)
            for sw in stop_words:
                if sw and sw in joined:
                    joined = joined.split(sw, 1)[0]
            answer_pieces = [joined]

        prompt_tokens = max(16, sum(len(str(m.get("content", ""))) for m in messages) // 2)
        completion_tokens = 48 if enable_thinking else 32
        prefill_s = 0.24
        decode_s = 1.42
        decode_tok_per_s = round(completion_tokens / decode_s, 2)
        prefill_tok_per_s = round(prompt_tokens / prefill_s, 2)
        cmpl_id = f"chatcmpl-mock-{int(time.time() * 1000)}"
        created = int(time.time())

        if not stream:
            msg_obj: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(answer_pieces),
            }
            if enable_thinking:
                msg_obj["reasoning_content"] = "".join(reasoning_pieces)
            if want_tool:
                msg_obj["tool_calls"] = [
                    {
                        "id": "call_mock_1",
                        "type": "function",
                        "function": {"name": "get_expert_stats", "arguments": '{"layer": 12}'},
                    }
                ]
            resp = {
                "id": cmpl_id,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "message": msg_obj, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "timings": {
                    "prefill_s": prefill_s,
                    "decode_s": decode_s,
                    "decode_tok_per_s": decode_tok_per_s,
                },
            }
            with STATE_LOCK:
                REQ_COUNT += 1
                LRU_HITS += 38
                PREFETCH_HITS += 6
                DEMAND_MISSES += 4
                LAST_REQUEST.update(
                    {
                        "endpoint": "/v1/chat/completions",
                        "finished_at": int(time.time()),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "finish_reason": "stop",
                        "prefill_s": prefill_s,
                        "prefill_tok_per_s": prefill_tok_per_s,
                        "decode_s": decode_s,
                        "decode_tok_per_s": decode_tok_per_s,
                        "request_s": round(time.perf_counter() - t_req0, 3),
                    }
                )
            self._send_json(200, resp)
            return

        _set_busy(True)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            def send_sse(obj: dict[str, Any]) -> None:
                line = f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")
                self.wfile.write(line)
                self.wfile.flush()

            time.sleep(0.04)
            if enable_thinking:
                for rp in reasoning_pieces:
                    send_sse(
                        {
                            "id": cmpl_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [
                                {"index": 0, "delta": {"reasoning_content": rp}, "finish_reason": None}
                            ],
                        }
                    )
                    time.sleep(0.03)

            if want_tool:
                send_sse(
                    {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_mock_1",
                                            "type": "function",
                                            "function": {
                                                "name": "get_expert_stats",
                                                "arguments": '{"layer": 12, "top_k": 8}',
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                time.sleep(0.02)

            for ap in answer_pieces:
                send_sse(
                    {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {"content": ap}, "finish_reason": None}],
                    }
                )
                time.sleep(0.03)

            req_s = round(time.perf_counter() - t_req0, 3)
            with STATE_LOCK:
                REQ_COUNT += 1
                LRU_HITS += 38
                PREFETCH_HITS += 6
                DEMAND_MISSES += 4
                DEMAND_READ_BYTES += 16_777_216
                PREFETCH_READ_BYTES += 25_165_824
                LAST_REQUEST.update(
                    {
                        "endpoint": "/v1/chat/completions",
                        "finished_at": int(time.time()),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "finish_reason": "stop",
                        "prefill_s": prefill_s,
                        "prefill_tok_per_s": prefill_tok_per_s,
                        "decode_s": decode_s,
                        "decode_tok_per_s": decode_tok_per_s,
                        "request_s": req_s,
                    }
                )

            send_sse(
                {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    "timings": {
                        "prefill_s": prefill_s,
                        "decode_s": decode_s,
                        "decode_tok_per_s": decode_tok_per_s,
                    },
                }
            )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            _set_busy(False)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), MockHandler)
    print(f"Mock server listening on http://{HOST}:{PORT}/ui/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
