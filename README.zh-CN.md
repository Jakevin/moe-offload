[English](README.md) | [繁體中文](README.zh-TW.md) | **简体中文**

# moe-offload

面向 Apple Silicon 上混合专家（Mixture-of-Experts, MoE）大语言模型的实验性专家卸载加载器。始终会用到的权重常驻于统一内存；路由专家存放在 SSD 上，按需读入固定预算、逐层的 LRU 缓存。还可选用基于路由器的预取，提前读取下一层的专家。

这是研究代码，仅在下方列出的模型和机器上测试过。

## 内容

| 路径 | 说明 |
|---|---|
| `mlx-qwen38/` | Qwen3.8-Flash-Next（MLX GSQ-RCO Q2 重新打包）的 MLX 加载器：`loader.py`、`bench.py`、`profile_run.py`、`server.py`（OpenAI 兼容服务器）、`run.sh`（RSS／swap 监控程序）、`expert_io.c`（原生 pthread pread 线程池，通过 ctypes 加载） |
| `mlx-qwen38/trace/` | 路由追踪器与分析（逐层 LRU 命中模拟、下一层路由器预测召回率） |
| `llamacpp-bolt/` | 用于 NVIDIA Nemotron 3.5 Bolt 74B-A9B GGUF 的 llama.cpp 专家缓存：`llama-moe-cache.patch`、`moe_loader.cpp`、`route_trace.cpp`、`analyze_routes.py`、`bench.py` |

## 实测结果（Mac mini M4 Pro，64 GB，内置 SSD）

在测试提示词上，所有使用缓存的运行所产生的贪心解码输出都与参考运行完全一致。除下方 16 GB Mac 的数据行外，每次运行的 swap 增长均为 0。

**Qwen3.8-Flash-Next（125B MoE，每层 512 个专家，top-10），MLX。** 8 个提示词（代码／中文／数学／英文对话），每个 128 个 token。专家缓存 = 路由专家的 20%（约 7.4 GB），第 0 层固定常驻。

| 配置 | 解码 tok/s | 预填充 tok/s | 有效命中率 | 峰值 RSS |
|---|---|---|---|---|
| 参考追踪器（28 层常驻） | 2.52 | 5.3 | 不适用 | 24.9 GiB |
| v1，Python pread，无预取 | 8.51 | 14.8 | 84.3% | 11.6 GiB |
| v2，原生 pread，更少同步，无预取 | 9.17 | 18.9 | 84.3% | 10.3 GiB |
| v2 + 路由器预取 top-12，每层最多 3 个 | 9.73 | 19.4 | 94.7% | 8.9 GiB |

v1 每个 token 的性能剖析（共 118 ms）：GPU 27%、Python 与 `mx.eval` 同步 41%、SSD 读取 29%、记账开销 3%。该模型以 llama.cpp GGUF 完全常驻运行（不同配置）可达 24.6 tok/s。剩余差距主要来自 Python 的逐层同步。

下一层路由器预测（按第 l+1 层的方式混合 4 条 hyper-connection 流，再应用第 l+1 层的 norm 和路由器）：recall@10 / @16 / @24 = 83.6% / 94.1% / 96.7%。该数据来自 2 个提示词的短追踪。

**Nemotron 3.5 Bolt 74B-A9B（IQ2_XXS GGUF，每层 240 个专家，top-9），llama.cpp Metal。** 8 个提示词，每个 64 个 token。完整模型基线为 13.9 tok/s，占用 20.8 GiB。

| 缓存 | 解码 tok/s | 命中率 | 峰值 RSS |
|---|---|---|---|
| 4 GB | 11.07 | 86.1% | 9.9 GiB |
| 8 GB | 11.05 | 93.8% | 13.9 GiB |
| 12 GB | 11.28 | 97.0% | 17.9 GiB |

在 16 GB 的 Mac mini M4 上：4 GB 缓存以 4.39 tok/s 运行（命中率 86.1%，峰值 8.5 GiB）。8 GB 缓存发生颠簸（thrashing）而中止。

注意：原版 llama.cpp Metal 缺少 9 个激活专家所需的 `kernel_mul_mm_id_map0`，因此 Bolt 需要 `-ub 31`。本补丁添加了该实例化。

## 使用方法（概要）

- llama.cpp：`git checkout 620d2225cc436ab0b001e3200e590e3b286ef9ab && git apply llama-moe-cache.patch`，使用 `-DGGML_METAL=ON` 构建，然后运行 `make -C llamacpp-bolt`（需设置 `LLAMA_DIR`）。
- MLX：需要支持 `qwen4_exp` 的 MLX + mlx-vlm 版本。脚本默认使用 oMLX.app 自带的 Python。将 `QWEN38_MODEL` 设为模型目录、`MOE_WORKDIR` 设为工作目录（所有脚本都在此读写数据文件）。然后依次执行：
  1. **先追踪（必需）。** 运行 `trace/tracer.py`。它会运行参考模型，并将 `routes.jsonl`（每个 token 的路由和参考贪心解码 token）和 `trace_meta.json`（提示词和 token 数量）写入 `MOE_WORKDIR`。可选环境变量：`QWEN38_RESIDENT_LAYERS`（默认 28）、`QWEN38_DECODE_TOKENS`（默认 128）。
  2. **分析。** 运行 `trace/analyze.py`。它会读取 `routes.jsonl` 和 `trace_meta.json`，并输出 `analysis_summary.json`。
  3. **运行加载器。** `./run.sh <python> loader.py ...`。在默认的 `--mode verify` 下，`loader.py` 会读取 `routes.jsonl` 和 `trace_meta.json`，将输出与参考结果比对；若文件不存在则会失败。默认的 `--alloc-mode proportional` 会根据 `analysis_summary.json` 确定各层缓存大小（若无该文件，则改为平均分配）。常用参数：`--budget-gb` / `--budget-pct`、`--prefetch on|off`（默认 off）、`--prefetch-top-m`（默认 12）、`--prefetch-cap`（默认 3）、`--mode verify|single`、`--prompt`、`--decode-tokens`。

## OpenAI 兼容服务器（`mlx-qwen38/server.py`）

`server.py` 通过 `loader.OffloadEngine` 只加载一次模型，并提供 OpenAI 兼容 API（使用标准库 `http.server`，默认 `127.0.0.1:8090`）。**设计上一次只处理一个请求**（并发请求会排队等待）。

```bash
cd mlx-qwen38
./run.sh <python> -u server.py --host 127.0.0.1 --port 8090 \
  --budget-pct 20 --alloc-mode even --prefetch on --prefetch-top-m 12 --prefetch-cap 3
kill -TERM <server pid>
```

其他参数：`--budget-gb`、`--model-path`（或环境变量 `QWEN38_MODEL`）、`--model-id`、`--max-rss-gb`、`--max-swap-growth-gb`。用 `kill -TERM` 停止。

接口：
- `GET /health`、`GET /v1/models`
- `POST /v1/chat/completions`（支持 SSE 流式与非流式）
- `POST /v1/completions`
- `POST /v1/responses`（OpenAI Responses API；流式与非流式）。支持 function 工具调用（见英文 README 的 Tool calling 一节）；`previous_response_id` 或图片输入会返回 HTTP 400。

参数：`max_tokens` / `max_completion_tokens` / `max_output_tokens`（默认 4096，可用 `--max-tokens` 调整）、`temperature`（0 = 贪婪解码）、`top_p`、`stop`、`stream`、`seed`、`enable_thinking`（默认 false）。响应包含 `usage` 和 `timings`。

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

实测（Mac mini M4 Pro）：预热后解码约 10.2–10.7 tok/s；`temperature: 0` 的输出与直接运行 `loader.py` 完全一致。

## 许可证

Apache-2.0
