#!/bin/bash
# Paired benchmark forks: gate.sh OUTDIR CONTROL_ROOT CANDIDATE_ROOT [FORKS] [SCENARIOS]
#
# Each fork runs both arms in one fresh JVM each, alternating which arm goes first, so slow drift on the
# host (desktop GPU load, thermals) hits both arms equally. CONTROL_ROOT and CANDIDATE_ROOT are repo
# checkouts with `nativeBuild :benchmark:installDist` already done; they may be the same tree when the
# arms differ only by environment (CONTROL_ENV / CANDIDATE_ENV, e.g. "EUHEDRAL_LANE_PLACEMENT=FORK").
# Do not rebuild either root, or load the CPU/GPU, while this runs. Summarize with summarize.py OUTDIR.
set -u
OUT=$1; CONTROL=$2; CANDIDATE=$3; FORKS=${4:-6}; WARMUP=${WARMUP:-2}; ITERATIONS=${ITERATIONS:-3}
SCEN=${5:-'"decode:64:128","decode:1024:128","prefill:256","prefill:1024"'}
ARTIFACT=${EUHEDRAL_BENCH_ARTIFACT:-/mnt/shared/qwen38-quant/artifacts/qwen3_5_27b_compact_q3.edrl}
TOKENIZER=${EUHEDRAL_BENCH_TOKENIZER:-/mnt/shared/qwen38-quant/source/qwen}
mkdir -p "$OUT"
run_arm() {
  local arm=$1 root=$2 f=$3 extra
  extra=$([ "$arm" = control ] && echo "${CONTROL_ENV:-}" || echo "${CANDIDATE_ENV:-}")
  cat > "$OUT/cfg-$arm-$f.json" <<J
{"artifact":"$ARTIFACT","tokenizer":"$TOKENIZER","cudaLibrary":"$root/build/native/linux-x64/lib/libeuhedral_cuda.so","scenarios":[$SCEN],"warmup":$WARMUP,"iterations":$ITERATIONS,"output":"$OUT/$arm-$f.jsonl"}
J
  rm -f "$OUT/$arm-$f.jsonl"
  (cd "$root" && env $extra EUHEDRAL_CUDA_INCLUDE_DIR="$root/build/cuda-dev/linux-x64/include" \
     LD_LIBRARY_PATH="$root/build/cuda-dev/linux-x64/runtime" \
     benchmark/build/install/euhedral-inference-benchmark/bin/euhedral-inference-benchmark run \
     "$OUT/cfg-$arm-$f.json" --fork-id "$f" > "$OUT/log-$arm-$f.txt" 2>&1)
  echo "$arm fork $f exit $?" >> "$OUT/progress"
}
for f in $(seq 1 "$FORKS"); do
  if [ $((f % 2)) = 1 ]; then run_arm control "$CONTROL" "$f"; run_arm candidate "$CANDIDATE" "$f"
  else run_arm candidate "$CANDIDATE" "$f"; run_arm control "$CONTROL" "$f"; fi
done
echo DONE >> "$OUT/progress"
