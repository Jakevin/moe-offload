**English** | [繁體中文](README.zh-TW.md) | [简体中文](README.zh-CN.md)

# moe-offload

Experimental expert-offloading loaders for Mixture-of-Experts LLMs on Apple Silicon. The always-used weights stay in unified memory. Routed experts live on SSD and are read on demand into a fixed-budget, per-layer LRU cache. An optional router-based prefetch can read the next layer's experts ahead of time.

This is research code. It was tested only on the models and machines listed below.

## Contents

| Path | What |
|---|---|
| `mlx-qwen38/` | MLX loader for Qwen3.8-Flash-Next (MLX GSQ-RCO Q2 repack): `loader.py`, `bench.py`, `profile_run.py`, `server.py` (OpenAI-compatible server), `run.sh` (RSS/swap watchdog), `expert_io.c` (native pthread pread pool, loaded via ctypes) |
| `mlx-qwen38/cpp/` | C++ runner (MLX C++ port of the decode path), bit-exact vs the Python loader; see [`mlx-qwen38/cpp/README.md`](mlx-qwen38/cpp/README.md) |
| `mlx-qwen38/trace/` | Routing tracer + analysis (per-layer LRU hit simulation, next-layer router prediction recall) |
| `llamacpp-bolt/` | llama.cpp expert cache for NVIDIA Nemotron 3.5 Bolt 74B-A9B GGUF: `llama-moe-cache.patch`, `moe_loader.cpp`, `route_trace.cpp`, `analyze_routes.py`, `bench.py` |

## Measured results (Mac mini M4 Pro, 64 GB, internal SSD)

All cached runs produced greedy output identical to the reference run on the test prompts. Swap growth was 0 in every run except the 16 GB Mac rows below.

**Qwen3.8-Flash-Next (125B MoE, 512 experts/layer, top-10), MLX.** 8 prompts (code / Chinese / math / English chat), 128 tokens each. Expert cache = 20% of routed experts (about 7.4 GB), layer 0 pinned.

| Config | Decode tok/s | Prefill tok/s | Effective hit rate | Peak RSS |
|---|---|---|---|---|
| Reference tracer (28 layers resident) | 2.52 | 5.3 | n/a | 24.9 GiB |
| v1, Python pread, no prefetch | 8.51 | 14.8 | 84.3% | 11.6 GiB |
| v2, native pread, fewer syncs, no prefetch | 9.17 | 18.9 | 84.3% | 10.3 GiB |
| v2 + router prefetch top-12, max 3 per layer | 9.73 | 19.4 | 94.7% | 8.9 GiB |

Prefill tok/s above is `loader.py` on these short test prompts with one-shot (single-pass) prefill and no prompt cache. It does not apply to long prompts through `server.py`, which prefills in chunks (see Prompt cache below).

Profile of v1 per token (118 ms total): GPU 27%, Python and `mx.eval` syncs 41%, SSD reads 29%, bookkeeping 3%. A fully resident llama.cpp GGUF run of this model (different setup) reached 24.6 tok/s. The remaining gap is mostly Python per-layer synchronization.

Next-layer router prediction (mixing the 4 hyper-connection streams the way layer l+1 does, then applying layer l+1's norm and router): recall@10 / @16 / @24 = 83.6% / 94.1% / 96.7%. This comes from a short trace of 2 prompts.

**Nemotron 3.5 Bolt 74B-A9B (IQ2_XXS GGUF, 240 experts/layer, top-9), llama.cpp Metal.** 8 prompts, 64 tokens each. The full-model baseline was 13.9 tok/s at 20.8 GiB.

| Cache | Decode tok/s | Hit rate | Peak RSS |
|---|---|---|---|
| 4 GB | 11.07 | 86.1% | 9.9 GiB |
| 8 GB | 11.05 | 93.8% | 13.9 GiB |
| 12 GB | 11.28 | 97.0% | 17.9 GiB |

On a Mac mini M4 with 16 GB: the 4 GB cache ran at 4.39 tok/s (86.1% hit, 8.5 GiB peak). The 8 GB cache thrashed and was aborted.

Note: stock llama.cpp Metal lacks `kernel_mul_mm_id_map0` for 9 active experts, so Bolt needs `-ub 31`. The patch adds the instantiation.

## Usage (sketch)

- llama.cpp: `git checkout 620d2225cc436ab0b001e3200e590e3b286ef9ab && git apply llama-moe-cache.patch`, build with `-DGGML_METAL=ON`, then `make -C llamacpp-bolt` (set `LLAMA_DIR`).
- MLX: needs an MLX + mlx-vlm build with `qwen4_exp` support. The scripts default to oMLX.app's bundled Python. Set `QWEN38_MODEL` to the model directory and `MOE_WORKDIR` to a working directory (all scripts read and write their data files there). Then, in order:
  1. **Trace first (required).** Run `trace/tracer.py`. It runs the reference model and writes `routes.jsonl` (per-token routing and reference greedy tokens) and `trace_meta.json` (prompts and token counts) into `MOE_WORKDIR`. Optional env vars: `QWEN38_RESIDENT_LAYERS` (default 28), `QWEN38_DECODE_TOKENS` (default 128).
  2. **Analyze.** Run `trace/analyze.py`. It reads `routes.jsonl` and `trace_meta.json` and writes `analysis_summary.json`.
  3. **Run the loader.** `./run.sh <python> loader.py ...`. In the default `--mode verify`, `loader.py` reads `routes.jsonl` and `trace_meta.json` to compare its output with the reference, and fails if they are missing. The default `--alloc-mode proportional` uses `analysis_summary.json` to size each layer's cache (if the file is absent, it falls back to an even split). Useful flags: `--budget-gb` / `--budget-pct`, `--prefetch on|off` (default off), `--prefetch-top-m` (default 12), `--prefetch-cap` (default 3), `--mode verify|single`, `--prompt`, `--decode-tokens`.

## OpenAI-compatible server (`mlx-qwen38/server.py`)

`server.py` loads the model once via `loader.OffloadEngine` and serves an OpenAI-compatible API (stdlib `http.server`, default `127.0.0.1:8090`). Requests are processed **one at a time by design** (concurrent requests wait on a lock).

```bash
cd mlx-qwen38
./run.sh <python> -u server.py --host 127.0.0.1 --port 8090 \
  --budget-pct 20 --alloc-mode even --prefetch on --prefetch-top-m 12 --prefetch-cap 3
kill -TERM <server pid>
```

Other flags: `--max-tokens` (default output limit when a request omits it; default `-1` = unlimited, like llama.cpp and Ollama: generation stops at EOS / a stop string or when the context is full; `N` sets a fixed default), `--budget-gb`, `--model-path` (or `QWEN38_MODEL`), `--model-id`, `--max-rss-gb`, `--max-swap-growth-gb`. Stop with `kill -TERM`.

Endpoints:
- `GET /health`, `GET /v1/models`
- `POST /v1/chat/completions` (streaming SSE and non-streaming; function tools supported)
- `POST /v1/completions`
- `POST /v1/responses` (OpenAI Responses API; streaming and non-streaming). Function tools are supported (see Tool calling); `previous_response_id` and image inputs return HTTP 400.

Params: `max_tokens` / `max_completion_tokens` / `max_output_tokens` (default unlimited, capped at the remaining context; a request value, or `-1` for unlimited, overrides `--max-tokens`), `temperature` (0 = greedy), `top_p`, `stop`, `stream`, `seed`, `enable_thinking` (default false). Responses include `usage` and `timings`.

```bash
curl -s http://127.0.0.1:8090/health
curl -s http://127.0.0.1:8090/v1/models
curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"temperature":0}'
curl -N http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"stream":true}'
curl -s http://127.0.0.1:8090/v1/responses -H 'Content-Type: application/json' \
  -d '{"input":"Hello","max_output_tokens":64}'
```

Measured (Mac mini M4 Pro): ~10.2–10.7 tok/s warm decode; `temperature: 0` output is identical to running `loader.py` directly.

### Tool calling

Both `POST /v1/chat/completions` and `POST /v1/responses` support OpenAI-style function calling (`mlx-qwen38/tool_calls.py`, unit tests in `mlx-qwen38/test_tool_calls.py`, no model needed: `python -m unittest test_tool_calls -v`).

- **Model format**: the model's `chat_template.jinja` uses the Qwen3-Coder XML format. `tools` are passed to `apply_chat_template(tools=...)` and rendered exactly as the template does; the model replies with `<tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n</function>\n</tool_call>`. Parameter values are typed from each tool's JSON schema. Multiple calls, missing closing tags and a JSON `<tool_call>{"name":..,"arguments":..}</tool_call>` fallback are handled.
- **Chat Completions**: accepts `tools`, `tool_choice` (`auto`, `none`, `required`, or a named function), `parallel_tool_calls` (accepted, ignored); assistant messages with `tool_calls` and `role: "tool"` messages are accepted back. Returns `message.tool_calls` with `finish_reason: "tool_calls"`. Streaming sends text deltas before the call, then one `delta.tool_calls` chunk per call (full arguments), then `finish_reason: "tool_calls"` and `[DONE]`.
- **Responses**: accepts flat `{"type":"function","name",...}` tools, `function_call` and `function_call_output` input items (`reasoning` items skipped, `developer` mapped to `system`). Output contains an optional `message` item followed by `function_call` items (`call_id`, `name`, `arguments`). Streaming emits `response.output_item.added`, `response.function_call_arguments.delta`, `response.function_call_arguments.done`, `response.output_item.done`, then `response.completed`.
- **Output limit**: no default `max_tokens`, so long tool-call arguments are not cut off; generation ends at EOS or when prompt + output reach the model context (`max_position_embeddings`, 262144 here; `finish_reason: "length"` / Responses `incomplete`). Send `max_tokens` / `max_output_tokens`, or start the server with `--max-tokens N`, to bound a turn.
- **Generation thread**: all MLX prefill/decode runs on one dedicated worker thread (fixes `There is no Stream(gpu, N) in current thread` errors when consecutive long requests landed on different HTTP handler threads).

Limits:
- Still **one request at a time**.
- Tool-call arguments are emitted once at the end of generation, not token by token.
- `previous_response_id` is unsupported (send prior `function_call` / `function_call_output` items explicitly). Non-function tool types (e.g. `web_search`) are dropped.
- Thinking is off by default.
- Prompt cache is off by default (`--prompt-cache on`, see below). Prefill time depends on the mode. With one-shot (single-pass) prefill and no memory cap, agent clients' ~15–18k-token prompts took ~1.5–2 min (about 105 s) per turn on an M4 Pro. The server now prefills in chunks (`--prefill-chunk 2048`, the default), and each chunk re-reads the routed experts (about 35 s per pass), so long prompts are slower whenever the cache does not hit: with a 20 GB memory cap, a 14,937-token prompt took 315.7 s TTFT with no cache, 305.9 s on a cache miss and 63.9 s on a cache hit. Set client timeouts accordingly.

### Prompt cache (`--prompt-cache on`)

Single-session, in-memory cache (`prompt_cache.py`). It snapshots the full model state (linear-attention/GDN conv+SSM state, PLE window, QSA KV + indexer state, offsets, position state) plus next-token logits at reusable prefix points: the end of each prompt (= end of the last confirmed turn) and, once a system+tools prefix has been seen in two conversations, the end of the system message. Snapshots are keyed by exact token IDs plus model path, chat-template hash and control-vector settings, bounded by `--prompt-cache-max-snapshots` (4) and `--prompt-cache-max-gb` (1.5), LRU-evicted, and never shared with the live cache. Prefill runs in fixed chunks of `--prefill-chunk` tokens (default 2048, bounds memory) and only the last position goes through `lm_head`. Chunk size trades memory for speed: each chunk is another pass over the routed experts (about 35 s per pass), so smaller chunks lower peak memory but make long-prompt prefill slower, while larger chunks (up to one-shot prefill) need fewer passes but more memory.

Measured (M4 Pro 64 GB, 20 GB memory cap, 20% expert cache, warm, 3 interleaved rounds, all modes chunked with `--prefill-chunk 2048`; 14,937-token agent prompt, 80.9% reused from the previous 12,089-token turn):

| | TTFT |
|---|---|
| no cache (same 2048 chunking) | 315.7 s |
| cache miss (full chunked prefill) | 305.9 s (0.97x, CI 0.90–1.04) |
| cache hit (only the non-reused suffix is prefilled) | 63.9 s (**4.94x faster**, 95% CI 4.56–5.36) |

The no-cache and cache-miss times are slower than the one-shot ~105 s figure above (measured with no memory cap) because every 2048-token chunk re-reads the experts; only a cache hit avoids most of that cost.

Cached vs uncached runs with the same chunk boundaries are bit-exact: last-position logits, all 220 state arrays and 32 greedy tokens matched for prefix hits, identical-prompt hits (stored logits, no forward pass) and system-prompt hits. Changing the chunking itself (e.g. vs the original one-shot prefill) can change logits slightly (max |diff| 1.375 here; argmax and 32 greedy tokens unchanged). Peak process footprint 19.0 GB, swap +0. Unit tests: `python3.11 mlx-qwen38/test_prompt_cache.py`.

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages":[{"role":"user","content":"Weather in Taipei? Use the tool."}],
  "tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
  "temperature":0}'
```

Tested with opencode (`@ai-sdk/openai-compatible` provider, `baseURL: http://127.0.0.1:8090/v1`; run `opencode run ... < /dev/null`, or it waits on stdin) and Codex CLI (custom provider with `wire_api = "responses"`): both completed a read-a-file task via tool calls.

## Web UI (`mlx-qwen38/webui/`)

A lightweight local chat UI (vanilla HTML/CSS/JS, no build step, works offline) served by `server.py` itself:

```bash
cd mlx-qwen38
./run.sh <python> -u server.py --host 127.0.0.1 --port 8095 \
  --budget-pct 20 --alloc-mode even --prefetch on --prefetch-top-m 12 --prefetch-cap 3
open http://127.0.0.1:8095/ui/      # also served at /
kill -TERM <server pid>
```

Bind to `127.0.0.1` only (the server has no authentication). Pick any free port (the example avoids 11434, used by Ollama/oMLX).

Features: model picker (`/v1/models`); Chat / Thinking toggle (`enable_thinking`); streaming output, with the thinking text in a collapsible block; safe Markdown; stop (aborts the request), regenerate, copy; multiple conversations kept in browser `localStorage` (rename, delete, export JSON/Markdown, clear); settings for system prompt, `max_tokens`, `temperature`, `top_p`, `seed`, `stop`; per-reply prompt/output tokens, TTFT, prefill and decode tok/s, total time; a performance panel polling `/api/status` every 2 s (RSS, MLX active/peak, swap, expert-cache hit rate, last request, sparklines of the last 30 requests); light/dark theme.

Server additions used by the UI:
- `GET /`, `GET /ui/`, `GET /ui/<file>`: static files from `webui/` (whitelisted types, no path traversal).
- `GET /api/status`: busy flag, memory, expert-cache counters since start, last request timings, context length.
- `GET /healthz`: alias of `/health`.
- Thinking mode (`enable_thinking: true`, no tools): the text before `</think>` is returned as `delta.reasoning_content` (streaming) / `message.reasoning_content` (non-streaming), the rest as `content`. In greedy decoding, streaming and non-streaming returned identical text.

`webui/mock_server.py` is a stdlib-only fake API on `127.0.0.1:8097` for frontend-only development. Screenshot: `mlx-qwen38/webui/_test/webui_thinking_perf.png`. The UI's feature set was modelled on the Whallm Mac app's chat window (MIT); no Whallm code is used.

## C++ runner (`mlx-qwen38/cpp/`)

A C++ port of the decode path on the MLX C++ API, with the same SSD expert offload. Build and test:

```bash
cd mlx-qwen38/cpp && cmake -S . -B build && cmake --build build -j
./build/test_budget && ./build/test_io_sched && ./build/test_kernels   # run tests/gen_fixtures_v2.py first
QWEN38_MODEL=/path/to/model ./build/qwen38_decode --ids prompts.json --prompt-index 0 --n 128 --io exact --budget fixed --out out/run.json
```

Bit-exact gate (Mac mini M4 Pro 64 GB, 4 prompts × 128 tokens, 20% expert cache): token ids, every step's logits (raw bits) and the final cache state (122/122 tensors) are identical to the Python MLX loader. Decode is 9.0–9.6 tok/s at about 11.5 GB peak RSS. The P1 I/O scheduler, forced-miss stress and auto memory budget are also bit-identical. Dense attention only, so total length must be ≤ 2048 tokens. Details: [`mlx-qwen38/cpp/README.md`](mlx-qwen38/cpp/README.md).

## Experiments (not adopted)

- [`mlx-qwen38/experiments/a2-hot-expert-seeding/`](mlx-qwen38/experiments/a2-hot-expert-seeding/): A2, keeping the prompt's hot experts in the expert cache after prefill. Outputs stay bit-identical, but decode speedup was 1.008× (95% CI [0.967, 1.051]), below the pre-set gate, so it is **not adopted** and not merged into `loader.py`.

## License

Apache-2.0
