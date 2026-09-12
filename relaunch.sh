#!/usr/bin/env bash
# relaunch.sh — reproduce the 2026-09-11 standing stack (vllm-fn-ab1) exactly,
# with optional single-variable overrides for A/B arms.
#
# Standing recipe (verified via docker inspect vllm-fn-ab1, created 2026-09-11T18:41):
#   image vllm-skinny-tp1:v1-spinfix, name vllm-fn-ab1, mem 100g, ipc host,
#   ablit checkpoint, trimix_fill_65k draft vocab, MTP=3 argmax-reduction nodrop,
#   max-num-seqs 8, MNBT 4096, graphs [4..32], 17 mounts (patches + 50729/53388 + PLE).
#
# Overrides (env):
#   IMAGE              (default vllm-skinny-tp1:v1-spinfix)
#   EXTRA_VLLM_ARGS    (appended to the vllm CLI, e.g. "--moe-backend flashinfer_cutlass")
#   MTP_FILE           (host path to mounted mtp.py patch; default files/mtp_patched.py)
#   DRAFT_VOCAB        (host path; default ~/.cache/vllm/draft_vocab/trimix_fill_65k.txt)
#   CONTAINER_NAME     (default vllm-fn-ab1)
#   NO_RESTART         (1 = just print the command)
#   EXTRA_DOCKER       (extra docker args appended verbatim: -e/-v for patches)
#   DRAFT_TOPK         (sets VLLM_MTP_DRAFT_TOPK env in container, F4a-lite)
#   MTP_K              (overrides num_speculative_tokens; default 3)
set -uo pipefail
cd /home/nmt/flashnext-spark

IMAGE="${IMAGE:-vllm-skinny-tp1:v1-spinfix}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-fn-ab1}"
MTP_FILE="${MTP_FILE:-/home/nmt/flashnext-spark/files/mtp_patched.py}"
DRAFT_VOCAB="${DRAFT_VOCAB:-/home/nmt/.cache/vllm/draft_vocab/trimix_fill_65k.txt}"
EXTRA="${EXTRA_VLLM_ARGS:-}"
EXTRA_DOCKER="${EXTRA_DOCKER:-}"
if [[ -n "${DRAFT_TOPK:-}" ]]; then
  EXTRA_DOCKER="$EXTRA_DOCKER -e VLLM_MTP_DRAFT_TOPK=$DRAFT_TOPK"
fi

V=/usr/local/lib/python3.12/dist-packages/vllm

CMD=(docker run -d --name "$CONTAINER_NAME"
  --gpus all --network host --ipc host
  --cap-add SYS_NICE --cap-add SYS_PTRACE
  --ulimit memlock=-1 --ulimit stack=67108864
  --memory 100g --memory-swap 100g
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
  -e VLLM_PLE_CPU_OFFLOAD=1
  -e VLLM_PLE_PACKED_TABLE_DIR=/root/.cache/vllm/ple_cache/Mia-AiLab--Qwen3.8-Flash-Next-NVFP4
  -e VLLM_PLE_OFFLOAD_STEP_TIMEOUT=300
  -e VLLM_USE_V2_MODEL_RUNNER=1
  -e VLLM_MTP_DRAFT_VOCAB=/root/draft_vocab.txt
  -v "$DRAFT_VOCAB:/root/draft_vocab.txt:ro"
  -e HF_HOME=/root/.cache/huggingface
  -v /home/nmt/.cache/huggingface:/root/.cache/huggingface
  -v /home/nmt/.cache/vllm:/root/.cache/vllm
  -v /home/nmt/flashnext-spark/files/ple_layer_patched.py:$V/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro
  -v /home/nmt/flashnext-spark/files/modelopt_patched.py:$V/model_executor/layers/quantization/modelopt.py:ro
  -v /home/nmt/flashnext-spark/files/qsa_ops_patched.py:$V/models/qwen3_8_flash_next/nvidia/ops/qsa.py:ro
  -v /home/nmt/flashnext-spark/files/qsa_nvidia_patched.py:$V/models/qwen3_8_flash_next/nvidia/qsa.py:ro
  -v "$MTP_FILE:$V/models/qwen3_8_flash_next/nvidia/mtp.py:ro"
  -v /home/nmt/flashnext-spark/files/ple_offload/ple_offload_layer.py:$V/model_executor/layers/ple_offload_layer.py:ro
  -v /home/nmt/flashnext-spark/files/ple_offload/connector.py:$V/v1/ple_offload/connector.py:ro
  -v /home/nmt/flashnext-spark/files/ple_offload/worker.py:$V/v1/ple_offload/worker.py:ro
  -v /home/nmt/flashnext-spark/files/ple_offload/protocol.py:$V/v1/ple_offload/protocol.py:ro
  -v /home/nmt/flashnext-spark/files/mamba_utils_50729.py:$V/v1/worker/mamba_utils.py:ro
  -v /home/nmt/flashnext-spark/s3/out53388/vllm/config/speculative.py:$V/config/speculative.py:ro
  -v /home/nmt/flashnext-spark/s3/out53388/vllm/v1/core/kv_cache_utils.py:$V/v1/core/kv_cache_utils.py:ro
  -v /home/nmt/flashnext-spark/s3/out53388/vllm/v1/core/sched/scheduler.py:$V/v1/core/sched/scheduler.py:ro
  -v /home/nmt/flashnext-spark/s3/out53388/vllm/v1/core/single_type_kv_cache_manager.py:$V/v1/core/single_type_kv_cache_manager.py:ro
  $EXTRA_DOCKER
  "$IMAGE"
  drowzeys/keys-Qwen3.8-flash-next-ablit-Mia-Single-Spark-only
  --served-model-name qwen3.8-flash-next
  --tensor-parallel-size 1 --gpu-memory-utilization 0.786
  --max-num-seqs 8 --max-num-batched-tokens 4096 --max-model-len 262144
  --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16
  --load-format safetensors --safetensors-load-strategy lazy
  --enable-chunked-prefill --reasoning-parser qwen3
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --distributed-executor-backend mp
  --speculative-config '{"method":"mtp","num_speculative_tokens":'"${MTP_K:-3}"',"use_local_argmax_reduction":true,"disable_eagle_block_drop":true}'
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,12,16,20,24,28,32]}'
  $EXTRA
  --host 0.0.0.0 --port 8888)

if [[ "${NO_RESTART:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"; echo; exit 0
fi

echo "stopping standing container..."
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1
pkill -f 'VLLM::' 2>/dev/null
sleep 8
free -g | head -2
"${CMD[@]}"
echo "launched; polling health (expect ~10-12 min)..."
for i in $(seq 1 150); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8888/health 2>/dev/null)
  [[ "$code" == "200" ]] && { echo "HEALTHY after ~$((i*10))s"; exit 0; }
  sleep 10
done
echo "FATAL: not healthy after 25 min"; exit 1
