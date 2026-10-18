# Task: lightweight local Web UI for our OpenAI-compatible loader server

Write ONLY these files in this directory (/Volumes/1Tb/projects/moe-offload/qwen38/loader/webui):
- index.html
- app.js
- style.css
- mock_server.py
Do not edit any other file. Do not delete anything.

## Rules
- Plain HTML/CSS/vanilla JS. No build step, no npm, no CDN or external network (it must work offline).
- The page is served by our server at `GET /ui/` (index.html), with `/ui/app.js` and `/ui/style.css`. Use relative URLs for the API (same origin), e.g. `fetch("/v1/models")`.
- This is a feature reference only: do not copy any third-party code. Write it yourself.
- Do NOT start any model or LLM process.
- Do NOT bind ports 11434 or 8019.
- For testing, write `mock_server.py`: a stdlib-only fake server, bound to 127.0.0.1 port 8097. It serves `/ui/*` from this directory and fakes the API below, including streaming with reasoning_content. Test against it only; stop it when done.
- UI text in Traditional Chinese (繁體中文), with English technical terms where natural.

## API (already implemented by the real server)
- `GET /v1/models` returns `{data:[{id}]}`.
- `POST /v1/chat/completions`
  - Request body: `{model, messages, stream:true, max_tokens, temperature, top_p, seed, stop, enable_thinking, stream_options:{include_usage:true}}`.
  - Stream format: SSE lines `data: {...}`, ending with `data: [DONE]`.
  - Each chunk has `choices[0].delta.content`, or `choices[0].delta.reasoning_content` (thinking mode).
  - The final chunk has `finish_reason`, `usage {prompt_tokens, completion_tokens}` and `timings {prefill_s, decode_s, decode_tok_per_s}`.
  - Non-stream responses return `choices[0].message.{content, reasoning_content}`.
- `GET /api/status` returns:
  ```
  {busy, model,
   memory:{rss_gib, mlx_active_gib, mlx_peak_gib, swap_used_mb},
   expert_cache_since_start:{hit_rate, decode_requests, demand_misses, demand_read_bytes, prefetch_read_bytes},
   last_request:{prompt_tokens, completion_tokens, prefill_s, prefill_tok_per_s, decode_s, decode_tok_per_s, request_s, finish_reason},
   context_length}
  ```
- `GET /health` returns `{status, busy, rss_gib, mlx_active_gib, mlx_peak_gib}`.

## Features
1. **Toolbar:** model picker filled from /v1/models; Chat/Thinking mode toggle (Thinking sends `enable_thinking:true`); a live "Decode tok/s" readout.
2. **Streaming chat:**
   - Show the assistant reply as it streams, rendered as safe inline Markdown: bold, italic, inline code, code blocks, links. Escape HTML; never inject raw model HTML.
   - Show reasoning_content in a collapsible "思考過程" block above the answer. It stays expanded while streaming and collapses when the answer starts.
   - Show tool_calls (if present) as a small JSON block.
3. **Controls:** a Stop button that aborts the fetch via AbortController; Enter to send, Shift+Enter for a newline; a busy indicator while the server is busy.
4. **Conversation history:**
   - Multiple conversations, kept in localStorage: new, rename, delete, switch. The first user message becomes the default title.
   - Export a conversation as JSON or Markdown.
   - Clear the chat, with a confirmation.
   - Copy a message.
   - Regenerate the last answer.
5. **Settings panel:** system prompt, max_tokens, temperature, top_p, seed (blank = none), stop strings. Store these in localStorage.
6. **Per-message metrics** under each assistant reply: prompt tokens, output tokens, TTFT (measured client side), prefill tok/s, decode tok/s, total time.
7. **Performance panel** (side panel or drawer):
   - Poll /api/status every 2 s while visible. Show the memory values, expert cache hit rate, and last-request numbers.
   - Show small sparklines (canvas or SVG, hand-written) of decode tok/s and prefill tok/s over the last 30 requests.
   - Include a "clear history" button.
8. **Errors:** show HTTP and JSON errors inline. If the server is unreachable, show "伺服器未連線" and retry.
9. **Layout:** responsive, dark and light themes (prefers-color-scheme plus a toggle), accessible labels.

When done, write `webui/AGY_DONE.md`. It should list:
- the files written
- the features implemented
- how you tested against mock_server.py (exact commands)
- any limitations
