#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ICEFALL_ROOT="${ICEFALL_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/data1/${USER}}"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH="${ICEFALL_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TMPDIR="${TMPDIR:-${DATA_ROOT}/tmp}"
# NCCL defaults for single-node run without IB.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
mkdir -p "${TMPDIR}"

MANIFEST_DIR="${MANIFEST_DIR:-${DATA_ROOT}/data/french/manifests}"
TRAIN_CUTS_FILENAME="${TRAIN_CUTS_FILENAME:-msr_cuts_French_train.jsonl.gz}"
VALID_CUTS_FILENAME="${VALID_CUTS_FILENAME:-msr_cuts_French_valid.jsonl.gz}"
TEST_CUTS_FILENAME="${TEST_CUTS_FILENAME:-msr_cuts_French_test.jsonl.gz}"
MUSAN_CUTS_FILENAME="${MUSAN_CUTS_FILENAME:-musan_cuts_modify.jsonl.gz}"

# Preferred French BPE model path. If unavailable, pick first local ASR bpe.model.
BPE_MODEL="${BPE_MODEL:-${SCRIPT_DIR}/data/france/lang_bpe_2000/bpe.model}"
if [[ ! -f "${BPE_MODEL}" ]]; then
  FALLBACK_BPE="$(find "${SCRIPT_DIR}/data" -type f -path "*/lang_bpe_2000/bpe.model" | head -n 1 || true)"
  if [[ -n "${FALLBACK_BPE}" ]]; then
    BPE_MODEL="${FALLBACK_BPE}"
  fi
fi
if [[ ! -f "${BPE_MODEL}" ]]; then
  echo "BPE model not found. Set BPE_MODEL explicitly."
  exit 1
fi

WORLD_SIZE="${WORLD_SIZE:-8}"
DIST_BACKEND="${DIST_BACKEND:-nccl}"
MASTER_PORT="${MASTER_PORT:-12360}"
EXP_DIR="${EXP_DIR:-${DATA_ROOT}/experiments/zipformer/$(date +%Y%m%d)_france_onfly}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${EXP_DIR}/tensorboard}"
MAX_DURATION="${MAX_DURATION:-1000}"
MAX_TRAIN_CUT_DURATION="${MAX_TRAIN_CUT_DURATION:-30}"
NUM_BUCKETS="${NUM_BUCKETS:-60}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VALID_NUM_WORKERS="${VALID_NUM_WORKERS:-4}"
VALID_INTERVAL="${VALID_INTERVAL:-3000}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
BASE_LR="${BASE_LR:-0.045}"
NUM_EPOCHS="${NUM_EPOCHS:-80}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20000}"

# Usually disabled for online feature benchmarks to avoid extra startup overhead.
SKIP_OOM_SCAN="${SKIP_OOM_SCAN:-1}"
# Keep long runs alive by skipping only OOM batches.
SKIP_OOM_BATCH="${SKIP_OOM_BATCH:-1}"
# Disable startup-global cut filtering by default for large lazy manifests.
FILTER_CUTS="${FILTER_CUTS:-0}"
# Set to 1 only when musan_cuts file is present in MANIFEST_DIR.
ENABLE_MUSAN="${ENABLE_MUSAN:-0}"
# Compute WER during validation (recommended for online-training quality tracking).
COMPUTE_VALID_WER="${COMPUTE_VALID_WER:-1}"
# Limit WER validation cost; 0 means full validation set.
VALID_WER_MAX_BATCHES="${VALID_WER_MAX_BATCHES:-100}"
WER_LOWERCASE="${WER_LOWERCASE:-1}"

python ./zipformer/train.py \
  --world-size "${WORLD_SIZE}" \
  --dist-backend "${DIST_BACKEND}" \
  --master-port "${MASTER_PORT}" \
  --exp-dir "${EXP_DIR}" \
  --tensorboard-dir "${TENSORBOARD_DIR}" \
  --start-epoch 1 \
  --max-duration "${MAX_DURATION}" \
  --max-train-cut-duration "${MAX_TRAIN_CUT_DURATION}" \
  --num-buckets "${NUM_BUCKETS}" \
  --causal 1 \
  --use-fp16 1 \
  --num-workers "${NUM_WORKERS}" \
  --valid-num-workers "${VALID_NUM_WORKERS}" \
  --valid-interval "${VALID_INTERVAL}" \
  --log-interval "${LOG_INTERVAL}" \
  --base-lr "${BASE_LR}" \
  --max-train-steps "${MAX_TRAIN_STEPS}" \
  --enable-spec-aug 1 \
  --enable-musan "${ENABLE_MUSAN}" \
  --on-the-fly-feats True \
  --skip-oom-scan "${SKIP_OOM_SCAN}" \
  --skip-oom-batch "${SKIP_OOM_BATCH}" \
  --filter-cuts "${FILTER_CUTS}" \
  --compute-valid-wer "${COMPUTE_VALID_WER}" \
  --valid-wer-max-batches "${VALID_WER_MAX_BATCHES}" \
  --wer-lowercase "${WER_LOWERCASE}" \
  --num-encoder-layers 2,2,4,5,4,2 \
  --feedforward-dim 512,768,1536,2048,1536,768 \
  --encoder-dim 192,256,512,768,512,256 \
  --encoder-unmasked-dim 192,192,256,320,256,192 \
  --bpe-model "${BPE_MODEL}" \
  --manifest-dir "${MANIFEST_DIR}" \
  --train-cuts-filename "${TRAIN_CUTS_FILENAME}" \
  --valid-cuts-filename "${VALID_CUTS_FILENAME}" \
  --test-cuts-filename "${TEST_CUTS_FILENAME}" \
  --musan-cuts-filename "${MUSAN_CUTS_FILENAME}" \
  --num-epochs "${NUM_EPOCHS}"
