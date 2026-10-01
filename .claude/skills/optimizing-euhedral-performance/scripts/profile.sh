#!/bin/bash
# Nsight Systems trace of one scenario: profile.sh ROOT OUTDIR SCENARIO TAG [ITERATIONS]
# Writes OUTDIR/TAG.nsys-rep and OUTDIR/TAG.sqlite (analyze with trace_report.py). Set NSYS to the nsys
# binary. Absolute times under the profiler are inflated; compare traces only with each other.
ROOT=$1; OUT=$2; SC=$3; TAG=$4; IT=${5:-2}
NSYS=${NSYS:-/home/brandon/.hermes/workspace/ninfer-profile-20260921/tools/nsys/bin/nsys}
ARTIFACT=${EUHEDRAL_BENCH_ARTIFACT:-/mnt/shared/qwen38-quant/artifacts/qwen3_5_27b_compact_q3.edrl}
TOKENIZER=${EUHEDRAL_BENCH_TOKENIZER:-/mnt/shared/qwen38-quant/source/qwen}
mkdir -p "$OUT"
cat > "$OUT/cfg-$TAG.json" <<J
{"artifact":"$ARTIFACT","tokenizer":"$TOKENIZER","cudaLibrary":"$ROOT/build/native/linux-x64/lib/libeuhedral_cuda.so","scenarios":["$SC"],"warmup":1,"iterations":$IT,"output":"$OUT/$TAG.jsonl","overwrite":true}
J
cd "$ROOT" && env EUHEDRAL_CUDA_INCLUDE_DIR="$ROOT/build/cuda-dev/linux-x64/include" \
  LD_LIBRARY_PATH="$ROOT/build/cuda-dev/linux-x64/runtime" \
  "$NSYS" profile -o "$OUT/$TAG" --force-overwrite true --trace cuda,nvtx --sample none --cpuctxsw none \
  benchmark/build/install/euhedral-inference-benchmark/bin/euhedral-inference-benchmark run "$OUT/cfg-$TAG.json" \
  > "$OUT/$TAG.log" 2>&1
echo "$TAG exit $?"
"$NSYS" export --type sqlite --force-overwrite true -o "$OUT/$TAG.sqlite" "$OUT/$TAG.nsys-rep" > /dev/null 2>&1
