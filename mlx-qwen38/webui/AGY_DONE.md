# AGY_DONE — Local Web UI for OpenAI-Compatible Loader Server

## 1. Files Written
All files were created strictly inside `/Volumes/1Tb/projects/moe-offload/qwen38/loader/webui`:
- `index.html` — Single-page offline HTML structure served at `/ui/` (with `/ui/style.css` and `/ui/app.js`), using Traditional Chinese (繁體中文) labels and accessible ARIA attributes.
- `style.css` — Responsive stylesheet supporting light/dark themes (`prefers-color-scheme: dark` + manual `[data-theme]` toggle), collapsible reasoning blocks, code blocks, message metrics pills, and side drawers.
- `app.js` — Vanilla JavaScript application implementing SSE streaming, thinking mode toggle, safe Markdown rendering, `localStorage` persistence, and real-time `/api/status` polling with hand-written SVG sparklines.
- `mock_server.py` — Stdlib-only Python fake server bound to `127.0.0.1:8097` serving `/ui/*`, `/v1/models`, `/v1/chat/completions` (streaming SSE & non-streaming, `reasoning_content`, `tool_calls`, `usage`, `timings`), `/api/status`, and `/health`.
- `AGY_DONE.md` — This completion summary.

## 2. Features Implemented
1. **Toolbar**:
   - Model selector populated dynamically from `GET /v1/models`.
   - Mode toggle between `一般對話 (Chat)` (`enable_thinking: false`) and `思考模式 (Thinking)` (`enable_thinking: true`).
   - Live `Decode tok/s` readout and server `忙碌中 / 閒置` (Busy / Idle) status badge.
2. **Streaming Chat (`POST /v1/chat/completions`)**:
   - Streams SSE `data: {...}` lines until `data: [DONE]`.
   - Safe Markdown renderer: escapes all raw HTML (`&`, `<`, `>`, `"`, `'`) before formatting fenced code blocks (` ```lang `), inline code (`` ` ``), bold (`**`), italic (`*`), and safe HTTP/HTTPS/relative links.
   - Collapsible `思考過程 (reasoning_content)` `<details>` block above the assistant reply that stays expanded while reasoning tokens stream and automatically collapses once answer `content` begins streaming.
   - Displays `tool_calls` (when present) in a formatted JSON block.
3. **Controls**:
   - `停止生成` (Stop) button wired to `AbortController.abort()`.
   - `Enter` to send, `Shift+Enter` for newline (with `e.isComposing` guard for CJK IME input).
   - Busy indicator updated both during active streaming and via `/api/status`.
4. **Conversation History**:
   - Multiple conversations persisted in `localStorage` (`新增對話`, `重新命名`, `刪除對話`, switching).
   - First user message automatically becomes the default conversation title.
   - Export active conversation as `.json` (`匯出 JSON`) or `.md` (`匯出 MD`).
   - `清空對話` (Clear chat) with confirmation modal.
   - Per-message `複製` (Copy) button and `↻ 重新生成` (Regenerate last answer) button.
5. **Settings Panel**:
   - Configurable `System Prompt`, `Max Tokens`, `Temperature`, `Top P`, `Seed` (blank = omitted), and comma-separated `Stop Strings`, persisted in `localStorage` with a reset-to-defaults button.
6. **Per-Message Metrics**:
   - Displayed under every assistant reply: Prompt tokens (`輸入`), Output tokens (`輸出`), client-measured `TTFT (ms)`, `Prefill tok/s`, `Decode tok/s`, and `總耗時 (s)`.
7. **Performance Panel**:
   - Polls `GET /api/status` every 2 seconds while visible (`visibilityState` & drawer visibility aware).
   - Displays memory metrics (`rss_gib`, `mlx_active_gib`, `mlx_peak_gib`, `swap_used_mb`, `context_length`), expert cache metrics (`hit_rate`, `decode_requests`, `demand_misses`, `demand_read_bytes`, `prefetch_read_bytes`), and `last_request` metrics.
   - Hand-written SVG sparklines (polygon area + polyline + data points) for `Decode tok/s` and `Prefill tok/s` over the last 30 requests, plus a `清除紀錄` (Clear history) button.
8. **Error Handling**:
   - Inline error alert box for HTTP and JSON/SSE errors.
   - Top offline banner (`伺服器未連線`) with a `重試` (Retry) button when the server is unreachable.
9. **Responsive Layout & Theme**:
   - Desktop multi-column layout with responsive drawers for tablet/mobile viewports, dark/light theme support (`prefers-color-scheme` + manual toggle), and accessible labels.

## 3. How Tested Against `mock_server.py` (Exact Commands)
Syntax checks, static asset checks, endpoint checks, streaming SSE (`reasoning_content`, `content`, `tool_calls`, `usage`, `timings`), and HTTP error handling were verified against `mock_server.py` on `127.0.0.1:8097` and stopped cleanly:

```bash
node --check app.js && python3 -m py_compile mock_server.py && python3 -c '
import subprocess, time, urllib.request, urllib.error, json

proc = subprocess.Popen(["python3", "mock_server.py"])
try:
    time.sleep(0.4)
    base = "http://127.0.0.1:8097"

    # 1. Static files
    for path, ctype in [("/ui/", "text/html"), ("/ui/app.js", "application/javascript"), ("/ui/style.css", "text/css")]:
        with urllib.request.urlopen(base + path) as r:
            assert r.status == 200, f"{path} status {r.status}"
            hdr = r.headers.get("Content-Type", "")
            assert ctype in hdr, f"{path} content-type {hdr}"
            body = r.read()
            assert len(body) > 100

    # 2. Models
    with urllib.request.urlopen(base + "/v1/models") as r:
        models = json.loads(r.read().decode())
        assert "data" in models and len(models["data"]) >= 1

    # 3. Health & Status
    with urllib.request.urlopen(base + "/health") as r:
        h = json.loads(r.read().decode())
        assert h["status"] == "ok" and "rss_gib" in h

    with urllib.request.urlopen(base + "/api/status") as r:
        st = json.loads(r.read().decode())
        assert "memory" in st and "expert_cache_since_start" in st and "last_request" in st and "context_length" in st

    # 4. Streaming chat with enable_thinking: true & tool_calls
    req_data = json.dumps({
        "model": models["data"][0]["id"],
        "messages": [{"role": "user", "content": "請測試思考模式與 [TOOL] 呼叫"}],
        "stream": True,
        "max_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "enable_thinking": True,
        "stream_options": {"include_usage": True}
    }).encode("utf-8")
    req = urllib.request.Request(base + "/v1/chat/completions", data=req_data, headers={"Content-Type": "application/json"})
    chunks = []
    saw_done = False
    with urllib.request.urlopen(req) as r:
        for raw_line in r:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                saw_done = True
                break
            chunks.append(json.loads(payload))
    assert saw_done, "Expected data: [DONE]"
    assert any(c["choices"][0]["delta"].get("reasoning_content") for c in chunks), "Expected reasoning_content chunks"
    assert any(c["choices"][0]["delta"].get("content") for c in chunks), "Expected content chunks"
    assert any(c["choices"][0]["delta"].get("tool_calls") for c in chunks), "Expected tool_calls chunks"
    assert "usage" in chunks[-1] and "timings" in chunks[-1], "Expected usage and timings on final chunk"

    # 5. Error handling test
    err_req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps({"model": "test", "messages": [{"role": "user", "content": "[ERROR]"}]}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(err_req)
        assert False, "Expected HTTP 500"
    except urllib.error.HTTPError as e:
        assert e.code == 500
        err_body = json.loads(e.read().decode())
        assert "error" in err_body

    print("ALL MOCK SERVER & STATIC ASSET TESTS PASSED!")
finally:
    proc.terminate()
    proc.wait(timeout=5)
'
```

And DOM ID cross-check + port cleanup verification:
```bash
python3 -c '
import re
html = open("index.html").read()
js = open("app.js").read()
html_ids = set(re.findall(r"id=\"([^\"]+)\"", html))
js_ids = set(re.findall(r"getElementById\(\"([^\"]+)\"\)", js))
missing = js_ids - html_ids
assert not missing, f"Missing IDs in index.html: {missing}"
print(f"Verified {len(js_ids)} DOM element IDs match between app.js and index.html.")
' && ! lsof -i :8097
```

## 4. Limitations
- **Lightweight Inline Markdown Parser**: Supports fenced code blocks, inline code, bold, italic, paragraphs/line breaks, and safe links without external dependencies; complex nested Markdown tables or LaTeX math expressions are rendered as plain escaped text.
- **Browser Storage Quota**: Conversations and 30-request sparkline metrics are stored in browser `localStorage` (typically ~5 MB per origin).
- **Single Active Generation**: In alignment with the local server's `GEN_LOCK`, the UI sends one generation request at a time.
