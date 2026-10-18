#!/usr/bin/env python3
"""Additive patch for server.py: /ui static Web UI, /healthz alias, /api/status, reasoning_content split
(thinking mode, no tools). Idempotent guard: refuses to run twice."""
import sys
from pathlib import Path
p = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent.parent / 'server.py')
s = open(p).read()
assert "_ThinkSplitter" not in s, "already patched"
def rep(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (a[:70], s.count(a)); s = s.replace(a, b)
rep('''def _encode_chat_messages(''', '''WEBUI_DIR = Path(__file__).resolve().parent / "webui"
_WEBUI_TYPES = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png", ".json": "application/json"}
LAST_REQUEST: dict[str, Any] = {}


def _req_enable_thinking(req: dict[str, Any]) -> bool:
    """Same precedence as _encode_chat_messages: enable_thinking, then chat_template_kwargs; default False."""
    ct_kwargs = req.get("chat_template_kwargs")
    if "enable_thinking" in req and req["enable_thinking"] is not None:
        return bool(req["enable_thinking"])
    if isinstance(ct_kwargs, dict) and ct_kwargs.get("enable_thinking") is not None:
        return bool(ct_kwargs["enable_thinking"])
    return False


class _ThinkSplitter:
    """Thinking mode: the prompt ends with '<think>\\\\n', so generated text is reasoning, '</think>', answer.
    Routes streamed text to reasoning_content until the closing tag, then to content (the reasoning_content
    field used by DeepSeek / vLLM / Open WebUI). Holds back a partial closing tag."""
    END = "</think>"

    def __init__(self) -> None:
        self.in_reasoning = True
        self.buf = ""
        self.answer_started = False

    def feed(self, s: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        if not self.in_reasoning:
            if not self.answer_started:
                s = s.lstrip("\\n")
                if not s:
                    return out
                self.answer_started = True
            return [("content", s)]
        self.buf += s
        if self.buf.startswith("<think>"):
            self.buf = self.buf[len("<think>"):]
        i = self.buf.find(self.END)
        if i >= 0:
            r, rest = self.buf[:i], self.buf[i + len(self.END):]
            self.buf = ""
            self.in_reasoning = False
            if r:
                out.append(("reasoning", r))
            return out + self.feed(rest)
        hold = 0
        for k in range(min(len(self.END) - 1, len(self.buf)), 0, -1):
            if self.buf.endswith(self.END[:k]):
                hold = k
                break
        emit = self.buf[: len(self.buf) - hold]
        self.buf = self.buf[len(emit):]
        if emit:
            out.append(("reasoning", emit))
        return out

    def flush(self) -> list[tuple[str, str]]:
        b, self.buf = self.buf, ""
        return [("reasoning" if self.in_reasoning else "content", b)] if b else []

    @staticmethod
    def split(text: str) -> tuple[str, str]:
        """Non-stream: (reasoning, content)."""
        t = text[len("<think>"):] if text.startswith("<think>") else text
        i = t.find(_ThinkSplitter.END)
        if i < 0:
            return t, ""
        return t[:i], t[i + len(_ThinkSplitter.END):].lstrip("\\n")


def _record_last_request(endpoint: str, t_req0: float | None, n_prompt: int, n_comp: int,
                         finish_reason: Any, timings: dict[str, Any], timings_raw: dict[str, Any]) -> None:
    pre = timings.get("prefill_s") or 0.0
    LAST_REQUEST.clear()
    LAST_REQUEST.update({
        "endpoint": endpoint, "finished_at": int(time.time()), "prompt_tokens": n_prompt,
        "completion_tokens": n_comp, "finish_reason": finish_reason, "prefill_s": pre,
        "prefill_tok_per_s": round(n_prompt / pre, 2) if pre else None,
        "decode_s": timings.get("decode_s"), "decode_tok_per_s": timings.get("decode_tok_per_s"),
        "request_s": round(time.perf_counter() - t_req0, 3) if t_req0 else None,
        "prompt_cache": timings_raw.get("prompt_cache"),
    })


def _encode_chat_messages(''')
rep('''    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/health":''', '''    def _send_static(self, rel: str) -> None:
        base = WEBUI_DIR.resolve()
        target = (base / rel).resolve()
        if base not in target.parents or not target.is_file() or target.suffix not in _WEBUI_TYPES:
            self._send_error_json(404, f"Unknown endpoint: {self.path}", "not_found_error")
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", _WEBUI_TYPES[target.suffix])
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
            self._send_static(path[len("/ui/"):])
            return
        if path == "/api/status":
            eng = self.server.engine
            try:
                lru_h, pref_h, dem_m, req_n, dem_b, pref_b = eng._snapshot_decode_totals()
                cache = {"decode_requests": req_n, "lru_hits": lru_h, "prefetch_hits": pref_h,
                         "demand_misses": dem_m, "hit_rate": round((lru_h + pref_h) / req_n, 4) if req_n else None,
                         "demand_read_bytes": dem_b, "prefetch_read_bytes": pref_b}
            except Exception as exc:  # noqa: BLE001
                cache = {"error": type(exc).__name__}
            self._send_json(200, {
                "status": "ok", "model": self.server.model_id, "busy": GEN_LOCK.locked(),
                "memory": {"rss_gib": round(loader.get_rss_kb() / (1024**2), 3),
                           "mlx_active_gib": round(mx.get_active_memory() / (1024**3), 3),
                           "mlx_peak_gib": round(mx.get_peak_memory() / (1024**3), 3),
                           "swap_used_mb": round(loader.get_swap_used_mb(), 1)},
                "expert_cache_since_start": cache,
                "last_request": LAST_REQUEST,
                "context_length": self._context_length(),
            })
            return
        if path in ("/health", "/healthz"):''')
rep('''        is_chat: bool,
        piece: str,
    ) -> None:''', '''        is_chat: bool,
        piece: str,
        kind: str | None = None,
    ) -> None:''')
rep('''        if is_chat:
            choice = {"index": 0, "delta": {"content": piece}, "finish_reason": None}
            obj = "chat.completion.chunk"''', '''        splitter = getattr(self, "_think", None)
        if is_chat and splitter is not None and kind is None:
            for k, t in splitter.feed(piece):
                if t:
                    self._emit_delta_chunk(cmpl_id, created, model_id, is_chat, t, kind=k)
            return
        if is_chat:
            key = "reasoning_content" if kind == "reasoning" else "content"
            choice = {"index": 0, "delta": {key: piece}, "finish_reason": None}
            obj = "chat.completion.chunk"''')
rep('''        n_prompt = len(prompt_ids)
        has_tools = bool(is_chat and tools)
''', '''        n_prompt = len(prompt_ids)
        has_tools = bool(is_chat and tools)
        # reasoning_content split only in thinking mode without tools (the tool parser owns the text then)
        self._think = _ThinkSplitter() if (is_chat and not has_tools and getattr(self, "_think_split_req", False)) else None
''')
rep('''    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]''', '''    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        self._t_req0 = time.perf_counter()
        self._think_split_req = False''')
rep('''            effective_tools: list[dict[str, Any]] = []
            if is_chat:
                messages = _normalize_messages(req.get("messages"))''', '''            effective_tools: list[dict[str, Any]] = []
            self._think_split_req = bool(is_chat and _req_enable_thinking(req))
            if is_chat:
                messages = _normalize_messages(req.get("messages"))''')
rep('''                    end_choice = (
                        {"index": 0, "delta": {}, "finish_reason": finish_reason}''', '''                    if self._think is not None:
                        for k, t in self._think.flush():
                            self._emit_delta_chunk(cmpl_id, created, model_id, is_chat, t, kind=k)
                    end_choice = (
                        {"index": 0, "delta": {}, "finish_reason": finish_reason}''')
rep('''                else:
                    msg_payload = {
                        "role": "assistant",
                        "content": content_text if content_text is not None else "",
                    }''', '''                else:
                    msg_payload = {
                        "role": "assistant",
                        "content": content_text if content_text is not None else "",
                    }
                    if self._think is not None:
                        r_txt, c_txt = _ThinkSplitter.split(content_text or "")
                        msg_payload["content"] = c_txt
                        msg_payload["reasoning_content"] = r_txt''')
rep('''                "decode_tok_per_s": timings_raw.get("decode_tok_per_s", 0.0),
            }
''', '''                "decode_tok_per_s": timings_raw.get("decode_tok_per_s", 0.0),
            }
            _record_last_request(self.path.split("?", 1)[0], getattr(self, "_t_req0", None), n_prompt, n_comp,
                                 locals().get("finish_reason") or locals().get("status"), timings, timings_raw)
''', cnt=2)
open(p, 'w').write(s); print("patched")
