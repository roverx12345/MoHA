#!/usr/bin/env bash
set -euo pipefail

# Select a checked GPU explicitly; keep server logs and model weights outside MoHA.
: "${MOHA_WHISPER_GPU:?Set MOHA_WHISPER_GPU to an available GPU index}"
case "$MOHA_WHISPER_GPU" in *[!0-9]*) exit 2 ;; esac
vllm_bin="${MOHA_WHISPER_VLLM:-/home/jianghan/miniconda3/envs/qwen35-vllm/bin/vllm}"
model_path="${MOHA_WHISPER_MODEL_PATH:-/home/jianghan/.cache/huggingface/hub/models--openai--whisper-large-v3-turbo/snapshots/6ce23f678cdbd6082c0e63d6202013f3624c1242}"
exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$MOHA_WHISPER_GPU" HF_HUB_OFFLINE=1 \
  VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONUNBUFFERED=1 \
  "$vllm_bin" serve "$model_path" --host 127.0.0.1 \
  --port "${MOHA_WHISPER_PORT:-8093}" --served-model-name whisper-large-v3-turbo \
  --dtype float16 --max-model-len 448 --enforce-eager \
  --gpu-memory-utilization 0.08 --max-num-seqs 2 --max-num-batched-tokens 2048
