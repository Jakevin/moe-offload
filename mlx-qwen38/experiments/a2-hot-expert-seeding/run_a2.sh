#!/bin/bash
# usage: run_a2.sh <loose|strict> <tag> <a2_trial.py args...>   idle-gated, frozen 80% budget, swap +8GB abort
cd "$(dirname "$0")"; mkdir -p logs out; NEED=$1; TAG=$2; shift 2
PY=/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11/bin/python3.11
export MOE_WORKDIR="$(pwd)/_wd"; export PYTHONPATH="/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages:/Applications/oMLX.app/Contents/Resources:/Applications/oMLX.app/Contents/Resources/omlx/patches/mlx_vlm_qwen4_exp_compat/vendor"
busy(){ pgrep -f "${A2_BUSY_PATTERN:-loader.py|server.py --|bench.py|llama-server|mlx_lm|a2_trial\.py}" >/dev/null; }
heavy(){ [ -n "$A2_HEAVY_PATTERN" ] && pgrep -f "$A2_HEAVY_PATTERN" >/dev/null; }
swap_mb(){ sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}'; }
avail_kb(){ vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} /Pages purgeable/{p=$3} END{gsub("\\.","",f);gsub("\\.","",i);gsub("\\.","",p); print int((f+i+p)*16384/1024)}'; }
ok=0; while [ $ok -lt 3 ]; do if busy || { [ "$NEED" = strict ] && heavy; }; then ok=0; else ok=$((ok+1)); fi; sleep 30; done
BASE=$(swap_mb); BUD=$(( $(avail_kb) * 8 / 10 )); BUD_GB=$(python3 -c "print(round($BUD*1024/1e9,1))")
echo "$(date '+%F %T') start $TAG swap_base=${BASE}MB budget=${BUD_GB}GB heavy_jobs=$( [ -n "$A2_HEAVY_PATTERN" ] && pgrep -f "$A2_HEAVY_PATTERN" | wc -l | tr -d ' ' || echo 0)"
$PY -u a2_trial.py "$@" --max-rss-gb $BUD_GB --out out/$TAG.jsonl > logs/$TAG.log 2>&1 & pid=$!
while kill -0 $pid 2>/dev/null; do r=$(ps -o rss= -p $pid | tr -d ' '); s=$(swap_mb)
  if [ "${r:-0}" -gt $BUD ] || [ $((s-BASE)) -gt 8192 ]; then echo "$(date '+%F %T') ABORT rss=$r swap=$s"; kill $pid; fi
  [ $((SECONDS % 60)) -lt 5 ] && echo "$(date "+%F %T") mon rss_kb=$r swap_mb=$s"; sleep 5; done
wait $pid; rc=$?; echo "$(date '+%F %T') end $TAG rc=$rc swap_end=$(swap_mb)MB"; exit $rc
