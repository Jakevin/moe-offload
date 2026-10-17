[English](README.md) | **繁體中文** | [简体中文](README.zh-CN.md)

# moe-offload

適用於 Apple Silicon 上混合專家（Mixture-of-Experts, MoE）大型語言模型的實驗性專家卸載載入器。每次都會用到的權重常駐於統一記憶體；路由專家存放在 SSD 上，依需求讀入固定預算、逐層的 LRU 快取。另可選用以路由器為基礎的預取，提前讀取下一層的專家。

這是研究用程式碼，僅在下方列出的模型與機器上測試過。

## 內容

| 路徑 | 說明 |
|---|---|
| `mlx-qwen38/` | Qwen3.8-Flash-Next（MLX GSQ-RCO Q2 重新打包）的 MLX 載入器：`loader.py`、`bench.py`、`profile_run.py`、`server.py`（OpenAI 相容伺服器）、`run.sh`（RSS／swap 監控程式）、`expert_io.c`（原生 pthread pread 執行緒池，透過 ctypes 載入） |
| `mlx-qwen38/trace/` | 路由追蹤器與分析（逐層 LRU 命中模擬、下一層路由器預測召回率） |
| `llamacpp-bolt/` | 用於 NVIDIA Nemotron 3.5 Bolt 74B-A9B GGUF 的 llama.cpp 專家快取：`llama-moe-cache.patch`、`moe_loader.cpp`、`route_trace.cpp`、`analyze_routes.py`、`bench.py` |

## 實測結果（Mac mini M4 Pro，64 GB，內建 SSD）

在測試提示詞上，所有使用快取的執行所產生的貪婪解碼輸出都與參考執行完全相同。除了下方 16 GB Mac 的資料列之外，每次執行的 swap 增長皆為 0。

**Qwen3.8-Flash-Next（125B MoE，每層 512 個專家，top-10），MLX。** 8 個提示詞（程式碼／中文／數學／英文對話），每個 128 個 token。專家快取 = 路由專家的 20%（約 7.4 GB），第 0 層固定常駐。

| 設定 | 解碼 tok/s | 預填 tok/s | 有效命中率 | 峰值 RSS |
|---|---|---|---|---|
| 參考追蹤器（28 層常駐） | 2.52 | 5.3 | 不適用 | 24.9 GiB |
| v1，Python pread，無預取 | 8.51 | 14.8 | 84.3% | 11.6 GiB |
| v2，原生 pread，較少同步，無預取 | 9.17 | 18.9 | 84.3% | 10.3 GiB |
| v2 + 路由器預取 top-12，每層最多 3 個 | 9.73 | 19.4 | 94.7% | 8.9 GiB |

v1 每個 token 的效能剖析（共 118 ms）：GPU 27%、Python 與 `mx.eval` 同步 41%、SSD 讀取 29%、簿記 3%。此模型以 llama.cpp GGUF 完全常駐執行（不同設定）可達 24.6 tok/s。剩餘差距主要來自 Python 的逐層同步。

下一層路由器預測（以第 l+1 層的方式混合 4 條 hyper-connection 串流，再套用第 l+1 層的 norm 與路由器）：recall@10 / @16 / @24 = 83.6% / 94.1% / 96.7%。此數據來自 2 個提示詞的短追蹤。

**Nemotron 3.5 Bolt 74B-A9B（IQ2_XXS GGUF，每層 240 個專家，top-9），llama.cpp Metal。** 8 個提示詞，每個 64 個 token。完整模型基準為 13.9 tok/s，佔用 20.8 GiB。

| 快取 | 解碼 tok/s | 命中率 | 峰值 RSS |
|---|---|---|---|
| 4 GB | 11.07 | 86.1% | 9.9 GiB |
| 8 GB | 11.05 | 93.8% | 13.9 GiB |
| 12 GB | 11.28 | 97.0% | 17.9 GiB |

在 16 GB 的 Mac mini M4 上：4 GB 快取以 4.39 tok/s 執行（命中率 86.1%，峰值 8.5 GiB）。8 GB 快取發生頻繁換頁（thrashing）而中止。

注意：原版 llama.cpp Metal 缺少 9 個活躍專家所需的 `kernel_mul_mm_id_map0`，因此 Bolt 需要 `-ub 31`。本修補檔加入了該實例化。

## 使用方式（概要）

- llama.cpp：`git checkout 620d2225cc436ab0b001e3200e590e3b286ef9ab && git apply llama-moe-cache.patch`，以 `-DGGML_METAL=ON` 建置，再執行 `make -C llamacpp-bolt`（需設定 `LLAMA_DIR`）。
- MLX：需要支援 `qwen4_exp` 的 MLX + mlx-vlm 版本。腳本預設使用 oMLX.app 內附的 Python。將 `QWEN38_MODEL` 設為模型目錄、`MOE_WORKDIR` 設為工作目錄（所有腳本都在此讀寫資料檔）。接著依序執行：
  1. **先追蹤（必要）。** 執行 `trace/tracer.py`。它會執行參考模型，並將 `routes.jsonl`（每個 token 的路由與參考貪婪解碼 token）和 `trace_meta.json`（提示詞與 token 數量）寫入 `MOE_WORKDIR`。可選環境變數：`QWEN38_RESIDENT_LAYERS`（預設 28）、`QWEN38_DECODE_TOKENS`（預設 128）。
  2. **分析。** 執行 `trace/analyze.py`。它會讀取 `routes.jsonl` 與 `trace_meta.json`，並寫出 `analysis_summary.json`。
  3. **執行載入器。** `./run.sh <python> loader.py ...`。在預設的 `--mode verify` 下，`loader.py` 會讀取 `routes.jsonl` 與 `trace_meta.json`，將輸出與參考結果比對；若檔案不存在則會失敗。預設的 `--alloc-mode proportional` 會依 `analysis_summary.json` 決定各層快取大小（若無此檔案，則改為平均分配）。常用參數：`--budget-gb` / `--budget-pct`、`--prefetch on|off`（預設 off）、`--prefetch-top-m`（預設 12）、`--prefetch-cap`（預設 3）、`--mode verify|single`、`--prompt`、`--decode-tokens`。

## OpenAI 相容伺服器（`mlx-qwen38/server.py`）

`server.py` 透過 `loader.OffloadEngine` 只載入一次模型，並提供 OpenAI 相容 API（使用標準函式庫 `http.server`，預設 `127.0.0.1:8090`）。**設計上一次只處理一個請求**（同時送出的請求會排隊等待）。

```bash
cd mlx-qwen38
./run.sh <python> -u server.py --host 127.0.0.1 --port 8090 \
  --budget-pct 20 --alloc-mode even --prefetch on --prefetch-top-m 12 --prefetch-cap 3
kill -TERM <server pid>
```

其他參數：`--budget-gb`、`--model-path`（或環境變數 `QWEN38_MODEL`）、`--model-id`、`--max-rss-gb`、`--max-swap-growth-gb`。以 `kill -TERM` 停止。

端點：
- `GET /health`、`GET /v1/models`
- `POST /v1/chat/completions`（支援 SSE 串流與非串流）
- `POST /v1/completions`
- `POST /v1/responses`（OpenAI Responses API；串流與非串流）。支援 function 工具呼叫（見英文 README 的 Tool calling 一節）；`previous_response_id` 或圖片輸入會回傳 HTTP 400。

參數：`max_tokens` / `max_output_tokens`、`temperature`（0 = 貪婪解碼）、`top_p`、`stop`、`stream`、`seed`、`enable_thinking`（預設 false）。回應包含 `usage` 與 `timings`。

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

實測（Mac mini M4 Pro）：暖機後解碼約 10.2–10.7 tok/s；`temperature: 0` 的輸出與直接執行 `loader.py` 完全相同。

## 授權

Apache-2.0
