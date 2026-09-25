#!/usr/bin/env bash
set -euo pipefail

# Select a checked GPU explicitly; keep server logs and model weights outside MoHA.
: "${MOHA_WHISPER_GPU:?Set MOHA_WHISPER_GPU to an available GPU index}"
case "$MOHA_WHISPER_GPU" in *[!0-9]*) exit 2 ;; esac
: "${MOHA_WHISPER_VLLM:?Set MOHA_WHISPER_VLLM to the vLLM executable}"
: "${MOHA_WHISPER_MODEL_PATH:?Set MOHA_WHISPER_MODEL_PATH to the Whisper model path}"
vllm_bin="$MOHA_WHISPER_VLLM"
model_path="$MOHA_WHISPER_MODEL_PATH"
exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$MOHA_WHISPER_GPU" HF_HUB_OFFLINE=1 \
  VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONUNBUFFERED=1 \
  "$vllm_bin" serve "$model_path" --host 127.0.0.1 \
  --port "${MOHA_WHISPER_PORT:-8093}" --served-model-name whisper-large-v3-turbo \
  --dtype float16 --max-model-len 448 --enforce-eager \
  --gpu-memory-utilization 0.08 --max-num-seqs 2 --max-num-batched-tokens 2048
