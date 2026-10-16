#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${MOE_WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$WORKDIR"

PYTHON_BIN="/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11/bin/python3.11"
export PYTHONPATH="/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages:/Applications/oMLX.app/Contents/Resources:/Applications/oMLX.app/Contents/Resources/omlx/patches/mlx_vlm_qwen4_exp_compat/vendor"

LOG_FILE="${WATCHDOG_LOG:-$WORKDIR/watchdog.log}"
STATS_FILE="${WATCHDOG_STATS:-$WORKDIR/watchdog_stats.json}"

# Hard limits:
# RSS < 38 GB (38,000,000 KiB = 38.0 GB decimal / 36.24 GiB)
MAX_RSS_KB=38000000
# Swap growth <= 8 GB (8192 MiB)
MAX_SWAP_DELTA_MB=8192

get_swap_used_mb() {
    sysctl vm.swapusage | awk '{
        for (i=1; i<=NF; i++) {
            if ($i == "used" && $(i+1) == "=") {
                val = $(i+2);
                gsub(/M/, "", val);
                print val;
                exit;
            }
        }
    }'
}

get_rss_kb() {
    local pid="$1"
    ps -o rss= -p "$pid" 2>/dev/null | awk '{print $1+0}' || echo 0
}

START_SWAP_MB="$(get_swap_used_mb)"
START_TIME="$(date +%s)"

echo "[watchdog] Starting at $(date -Iseconds)" | tee "$LOG_FILE"
echo "[watchdog] Initial swap used: ${START_SWAP_MB} MiB" | tee -a "$LOG_FILE"
echo "[watchdog] Limits: RSS <= ${MAX_RSS_KB} KiB (38 GB), Swap delta <= ${MAX_SWAP_DELTA_MB} MiB (8 GB)" | tee -a "$LOG_FILE"

if [ $# -eq 0 ]; then
    CMD=("$PYTHON_BIN" "$WORKDIR/loader.py")
else
    CMD=("$@")
fi

echo "[watchdog] Launching: ${CMD[*]}" | tee -a "$LOG_FILE"
"${CMD[@]}" &
CHILD_PID=$!

PEAK_RSS_KB=0
PEAK_SWAP_MB="$START_SWAP_MB"
PEAK_SWAP_DELTA_MB="0.00"
KILLED_BY_WATCHDOG=0
KILL_REASON=""

cleanup() {
    if kill -0 "$CHILD_PID" 2>/dev/null; then
        kill -TERM "$CHILD_PID" 2>/dev/null || true
    fi
}
trap cleanup INT TERM

while kill -0 "$CHILD_PID" 2>/dev/null; do
    RSS_KB="$(get_rss_kb "$CHILD_PID")"
    SWAP_MB="$(get_swap_used_mb)"
    if [ -n "$RSS_KB" ] && [ "$RSS_KB" -gt "$PEAK_RSS_KB" ]; then
        PEAK_RSS_KB="$RSS_KB"
    fi

    SWAP_DELTA_MB="$(awk -v cur="$SWAP_MB" -v start="$START_SWAP_MB" 'BEGIN { printf "%.2f", cur - start }')"
    IS_NEW_PEAK_SWAP="$(awk -v cur="$SWAP_MB" -v peak="$PEAK_SWAP_MB" 'BEGIN { print (cur > peak) ? 1 : 0 }')"
    if [ "$IS_NEW_PEAK_SWAP" -eq 1 ]; then
        PEAK_SWAP_MB="$SWAP_MB"
    fi
    IS_NEW_PEAK_DELTA="$(awk -v cur="$SWAP_DELTA_MB" -v peak="$PEAK_SWAP_DELTA_MB" 'BEGIN { print (cur > peak) ? 1 : 0 }')"
    if [ "$IS_NEW_PEAK_DELTA" -eq 1 ]; then
        PEAK_SWAP_DELTA_MB="$SWAP_DELTA_MB"
    fi

    if [ -n "$RSS_KB" ] && [ "$RSS_KB" -gt "$MAX_RSS_KB" ]; then
        KILLED_BY_WATCHDOG=1
        KILL_REASON="RSS ${RSS_KB} KiB exceeded limit ${MAX_RSS_KB} KiB (38 GB)"
        echo "[watchdog] KILLING PID ${CHILD_PID}: ${KILL_REASON}" | tee -a "$LOG_FILE"
        kill -9 "$CHILD_PID" 2>/dev/null || true
        break
    fi

    EXCEEDED_SWAP="$(awk -v delta="$SWAP_DELTA_MB" -v lim="$MAX_SWAP_DELTA_MB" 'BEGIN { print (delta > lim) ? 1 : 0 }')"
    if [ "$EXCEEDED_SWAP" -eq 1 ]; then
        KILLED_BY_WATCHDOG=1
        KILL_REASON="Swap delta ${SWAP_DELTA_MB} MiB exceeded limit ${MAX_SWAP_DELTA_MB} MiB (8 GB)"
        echo "[watchdog] KILLING PID ${CHILD_PID}: ${KILL_REASON}" | tee -a "$LOG_FILE"
        kill -9 "$CHILD_PID" 2>/dev/null || true
        break
    fi

    sleep 1
done

set +e
wait "$CHILD_PID"
EXIT_CODE=$?
set -e

END_SWAP_MB="$(get_swap_used_mb)"
END_TIME="$(date +%s)"
ELAPSED=$((END_TIME - START_TIME))
PEAK_RSS_GIB="$(awk -v kb="$PEAK_RSS_KB" 'BEGIN { printf "%.3f", kb / (1024.0 * 1024.0) }')"
PEAK_RSS_GB="$(awk -v kb="$PEAK_RSS_KB" 'BEGIN { printf "%.3f", (kb * 1024.0) / 1e9 }')"
END_SWAP_DELTA_MB="$(awk -v cur="$END_SWAP_MB" -v start="$START_SWAP_MB" 'BEGIN { printf "%.2f", cur - start }')"

echo "[watchdog] Finished at $(date -Iseconds) (elapsed ${ELAPSED}s, exit_code=${EXIT_CODE})" | tee -a "$LOG_FILE"
echo "[watchdog] Peak RSS: ${PEAK_RSS_KB} KiB (${PEAK_RSS_GIB} GiB / ${PEAK_RSS_GB} GB)" | tee -a "$LOG_FILE"
echo "[watchdog] Swap used: start=${START_SWAP_MB} MiB, end=${END_SWAP_MB} MiB (end delta=${END_SWAP_DELTA_MB} MiB, peak=${PEAK_SWAP_MB} MiB, peak delta=${PEAK_SWAP_DELTA_MB} MiB)" | tee -a "$LOG_FILE"

cat > "$STATS_FILE" <<EOF
{
  "exit_code": ${EXIT_CODE},
  "killed_by_watchdog": ${KILLED_BY_WATCHDOG},
  "kill_reason": "${KILL_REASON}",
  "elapsed_seconds": ${ELAPSED},
  "peak_rss_kb": ${PEAK_RSS_KB},
  "peak_rss_gib": ${PEAK_RSS_GIB},
  "peak_rss_gb": ${PEAK_RSS_GB},
  "start_swap_mb": ${START_SWAP_MB},
  "end_swap_mb": ${END_SWAP_MB},
  "peak_swap_mb": ${PEAK_SWAP_MB},
  "end_swap_delta_mb": ${END_SWAP_DELTA_MB},
  "peak_swap_delta_mb": ${PEAK_SWAP_DELTA_MB}
}
EOF

if [ "$KILLED_BY_WATCHDOG" -eq 1 ]; then
    exit 137
fi
exit "$EXIT_CODE"
