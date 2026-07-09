#!/usr/bin/env bash
set -euo pipefail

# Round 3: GQA oracle v3 with selector attention concentration signals.
#
# Usage:
#   bash scripts/run_gqa4000_attn_oracle.sh
#
# Optional environment variables:
#   N=4000
#   NUM_SHARDS=8
#   OUT_DIR=results_gqa4000_attn
#   HF_ENDPOINT=https://hf-mirror.com

N="${N:-4000}"
NUM_SHARDS="${NUM_SHARDS:-8}"
OUT_DIR="${OUT_DIR:-results_gqa4000_attn}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

export HF_ENDPOINT

echo "[round3] N=${N}"
echo "[round3] NUM_SHARDS=${NUM_SHARDS}"
echo "[round3] OUT_DIR=${OUT_DIR}"
echo "[round3] HF_ENDPOINT=${HF_ENDPOINT}"

mkdir -p "${OUT_DIR}"

if ! command -v python >/dev/null 2>&1; then
  echo "[round3][error] python not found. Did you activate conda env elastic?" >&2
  exit 1
fi

python - <<'PY'
import importlib.util
missing = [m for m in ["torch", "datasets", "transformers"] if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing packages: {missing}. Did you activate conda env elastic?")
PY

echo "[round3] launching oracle shards..."
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  log="${OUT_DIR}/logs_oracle_${i}.log"
  echo "[round3] shard ${i} -> ${log}"
  CUDA_VISIBLE_DEVICES="${i}" nohup python scripts/oracle_gqa.py \
    --n "${N}" --shard "${i}" --num-shards "${NUM_SHARDS}" \
    --out "${OUT_DIR}/oracle.json" \
    > "${log}" 2>&1 &
done

echo "[round3] launched. Waiting for all shards..."
wait

echo "[round3] all shard processes exited. Checking outputs..."
missing=0
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  shard_file="${OUT_DIR}/oracle.shard${i}.json"
  if [[ ! -s "${shard_file}" ]]; then
    echo "[round3][error] missing ${shard_file}" >&2
    missing=1
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  echo "[round3][error] one or more shards failed. Inspect ${OUT_DIR}/logs_oracle_*.log" >&2
  exit 1
fi

echo "[round3] running oracle analysis..."
python scripts/analyze_oracle.py --dir "${OUT_DIR}"

echo "[round3] running offline router analysis..."
python scripts/analyze_router_offline.py \
  --dir "${OUT_DIR}" \
  --out "${OUT_DIR}/router_offline.json" \
  --csv "${OUT_DIR}/router_labeled_samples.csv" \
  --md "${OUT_DIR}/router_offline.md"

echo "[round3] running train/test router search..."
python scripts/search_router_offline.py \
  --dir "${OUT_DIR}" \
  --out "${OUT_DIR}/router_search.json" \
  --md "${OUT_DIR}/router_search.md"

ARCHIVE="oracle_results_gqa${N}_attn.tar.gz"
echo "[round3] packing ${ARCHIVE}..."
tar czf "${ARCHIVE}" "${OUT_DIR}/"

echo "[round3] done."
echo "[round3] send back: ${ARCHIVE}"
