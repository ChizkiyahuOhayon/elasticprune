#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SEED="${SEED:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
SAVE_EVERY="${SAVE_EVERY:-10}"

case "${MODE}" in
  smoke)
    N="${N:-16}"
    GPU_IDS="${GPU_IDS:-0}"
    OUT_DIR="${OUT_DIR:-results_textvqa_external_smoke}"
    ;;
  full)
    N="${N:-5000}"
    GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
    OUT_DIR="${OUT_DIR:-results_textvqa_external_full}"
    ;;
  *)
    echo "usage: bash scripts/run_textvqa_external_oracle.sh [smoke|full]" >&2
    exit 2
    ;;
esac

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NUM_SHARDS="${#GPUS[@]}"
export HF_ENDPOINT

echo "[textvqa] mode=${MODE}"
echo "[textvqa] n=${N} seed=${SEED}"
echo "[textvqa] gpus=${GPU_IDS} shards=${NUM_SHARDS}"
echo "[textvqa] out=${OUT_DIR}"

python - <<'PY'
import importlib.util
required = ["torch", "datasets", "transformers", "lmms_eval"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing packages: {missing}. Activate the elastic environment first.")
PY

python -m elasticprune.capture_smoke_test
python scripts/textvqa_oracle_smoke_test.py
mkdir -p "${OUT_DIR}"

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPUS[${shard}]}"
  log="${OUT_DIR}/logs_oracle_${shard}.log"
  echo "[textvqa] shard ${shard} on GPU ${gpu} -> ${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/oracle_textvqa.py \
    --n "${N}" \
    --seed "${SEED}" \
    --shard "${shard}" \
    --num-shards "${NUM_SHARDS}" \
    --out "${OUT_DIR}/oracle.json" \
    --save-every "${SAVE_EVERY}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --resume \
    > "${log}" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "[textvqa][error] a shard failed; inspect ${OUT_DIR}/logs_oracle_*.log" >&2
  echo "[textvqa] rerun the same command to resume completed samples" >&2
  exit 1
fi

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  test -s "${OUT_DIR}/oracle.shard${shard}.json" || {
    echo "[textvqa][error] missing final shard ${shard}" >&2
    exit 1
  }
done

python scripts/analyze_textvqa_oracle.py --dir "${OUT_DIR}"

ARCHIVE="textvqa_external_${MODE}.tar.gz"
tar czf "${ARCHIVE}" "${OUT_DIR}"
echo "[textvqa] done: ${ARCHIVE}"
