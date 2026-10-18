# Loader Web UI (local only)

Static UI (index.html / app.js / style.css, vanilla JS, offline) served by server.py at `/ui/` (also `/`).
Feature reference: the Whallm Mac App chat window (MIT); no Whallm code copied.

## Start (one model process on the machine at a time)
```bash
cd mlx-qwen38
screen -dmS webui_test bash -c 'WATCHDOG_LOG=webui/_test/watchdog.log WATCHDOG_STATS=webui/_test/watchdog_stats.json \
  ./run.sh /Applications/oMLX.app/Contents/Resources/Python/cpython-3.11/bin/python3.11 -u server.py \
  --host 127.0.0.1 --port 8095 --budget-pct 20 --alloc-mode even --prefetch on --prefetch-top-m 12 --prefetch-cap 3 \
  --max-rss-gb 36 > webui/_test/server.log 2>&1'
open http://127.0.0.1:8095/ui/
```
Stop: `kill -TERM $(pgrep -f "server.py --host 127.0.0.1 --port 8095")`. Never use ports 11434 (oMLX) or 8019.

## New server endpoints (patch: webui/patch_server_webui.py, already applied to server.py; `python3 webui/patch_server_webui.py server.py` refuses to run twice)
- `GET /`, `/ui/`, `/ui/<file>`: static, path-safe whitelist
- `GET /api/status`: busy, memory, expert cache since start, last_request, context_length
- `GET /healthz`: alias of /health
- chat/completions with `enable_thinking`: `delta.reasoning_content` (stream) / `message.reasoning_content` (non-stream); disabled when tools are used

UI mock for frontend-only dev: `python3 webui/mock_server.py` (127.0.0.1:8097).
