#!/bin/bash
# GPT-OSS 20B architecture, random initialization; no pretrained weights loaded.
# Based on edo_train_llama33_70b_h200_8gpu.sh on Megatron-LM-Edgar-abci.
# Run from the repository root in the working Megatron environment:
#   bash examples/gptoss/edo_train_gptoss_20b_h200_8gpu.sh
# Synthetic benchmark: MOCK_DATA=1 bash examples/gptoss/edo_train_gptoss_20b_h200_8gpu.sh
# Command preview: MOCK_DATA=1 DRY_RUN=1 bash examples/gptoss/edo_train_gptoss_20b_h200_8gpu.sh
# Training uses BF16, not the released checkpoint's MXFP4 inference format.
# For roughly 21B parameters, FP32 master weights, Adam moments, and one
# FP32 gradient copy total roughly 336 GB (313 GiB), before other buffers.
# Full CPU optimizer offload retains substantial host/GPU memory headroom.
# Actual peak memory depends on runtime allocations; measure on the server.
# Defaults: 3 steps, 128-token synthetic sequences, no evaluation/checkpoints.
# Set MOCK_DATA=0 for arXiv; SAVE_CHECKPOINTS=1 to enable checkpoint writes.

python --version
python -c "import torch; print(torch.__version__)"


# Removing Checkpoint
echo removing checkpoints/....
rm -rf checkpoints/*

# Removing tensor-records
echo removing tensor-records/...
rm -rf tensorboard/*

# Removing chakras
echo removing chakras/...
rm -rf chakra/*


set -euo pipefail
die() { echo "ERROR: $*" >&2; exit 1; }

# 1. Run on one node with eight visible H200 GPUs. One torchrun agent launches eight GPU
# processes. --standalone chooses a local rendezvous endpoint automatically.
REPO_DIR=${REPO_DIR:-$PWD}
cd "$REPO_DIR"
REPO_DIR=$PWD
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
[[ -f pretrain_gpt.py ]] || die "Run from the Megatron-LM repository root"
NUM_NODES=1
GPUS_PER_NODE=8
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
[[ "$PYTHON_BIN" = /* && -x "$PYTHON_BIN" ]] || die "PYTHON_BIN must be an absolute executable path"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}

# 2. Four pipeline stages each hold 6 layers and use two GPUs.
# Attention uses TP=2; experts use EP=2, ETP=1 (16 experts per local layer).
# TP and EP reuse each stage's two GPUs; DP=1. Parameter memory per GPU
# remains similar; short sequences reduce activations and dispatch buffers.
# CPU optimizer offload creates memory headroom for full training.
TP=2
PP=4
EP=2
ETP=1
DP=$((WORLD_SIZE / (TP * PP)))
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-4}
SEQ_LENGTH=${SEQ_LENGTH:-128}
TRAIN_ITERS=${TRAIN_ITERS:-3}
WARMUP_ITERS=${WARMUP_ITERS:-0}
SAVE_INTERVAL=${SAVE_INTERVAL:-100}
SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-0}
OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-1.0}
[[ "$OPTIMIZER_OFFLOAD_FRACTION" =~ ^(0\.[0-9]*[1-9][0-9]*|1(\.0+)?)$ ]] || die "OPTIMIZER_OFFLOAD_FRACTION must be >0 and <=1"
MOCK_DATA=${MOCK_DATA:-1}
DRY_RUN=${DRY_RUN:-0}
for name in MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE SEQ_LENGTH TRAIN_ITERS SAVE_INTERVAL OMP_NUM_THREADS; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer"
done
[[ "$WARMUP_ITERS" =~ ^(0|[1-9][0-9]*)$ ]] || die "WARMUP_ITERS must be a nonnegative integer"
[[ "$MOCK_DATA" =~ ^[01]$ && "$DRY_RUN" =~ ^[01]$ && "$SAVE_CHECKPOINTS" =~ ^[01]$ ]] || die "MOCK_DATA, DRY_RUN and SAVE_CHECKPOINTS must be 0 or 1"
(( SEQ_LENGTH <= 131072 && SEQ_LENGTH % TP == 0 )) || die "SEQ_LENGTH must be divisible by $TP and <=131072"
(( WARMUP_ITERS < TRAIN_ITERS )) || die "WARMUP_ITERS must be less than TRAIN_ITERS"
(( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DP) == 0 )) || die "Global batch must be divisible by micro batch times DP"
NUM_MICROBATCHES=$((GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * DP)))

# 3. Use persistent storage for datasets, logs, and the large checkpoints.
# A new output directory prevents accidental reuse/overwriting of another run.
RUN_DIR=${RUN_DIR:-$REPO_DIR/runs/gptoss_20b_h200_8gpu/interactive-$(date +%Y%m%d-%H%M%S)-$$}
[[ "$RUN_DIR" = /* ]] || RUN_DIR="$REPO_DIR/$RUN_DIR"
[[ ! -e "$RUN_DIR" ]] || die "RUN_DIR already exists; choose a fresh directory"
# Resume only a checkpoint from this 20B architecture, never a 120B checkpoint.
RESUME_FROM=${RESUME_FROM:-}
CHECKPOINT_ARGS=()
if [[ -n "$RESUME_FROM" ]]; then
    [[ "$RESUME_FROM" = /* ]] || RESUME_FROM="$REPO_DIR/$RESUME_FROM"
    [[ -f "$RESUME_FROM/latest_checkpointed_iteration.txt" ]] || die "RESUME_FROM is not a Megatron checkpoint directory"
    CHECKPOINT_ARGS+=(--load "$RESUME_FROM" --exit-on-missing-checkpoint)
fi

# 4. Reuse the GPT-OSS tokenizer and arXiv files prepared for 120B.
# GPT-2 and Llama token IDs remain incompatible; no reprocessing is needed
# when switching between GPT-OSS 120B and 20B.
DATA_ARGS=(--split 949,50,1 --num-workers 1 --no-create-attention-mask-in-dataloader)
if [[ "$MOCK_DATA" == 1 ]]; then
    DATA_ARGS+=(--mock-data --tokenizer-type NullTokenizer --vocab-size 201088)
else
    TOKENIZER_MODEL=${TOKENIZER_MODEL:-$REPO_DIR/dataset/gptoss-tokenizer}
    DATA_PATH=${DATA_PATH:-$REPO_DIR/dataset/arxiv_gptoss_text_document}
    [[ "$TOKENIZER_MODEL" = /* ]] || TOKENIZER_MODEL="$REPO_DIR/$TOKENIZER_MODEL"
    [[ "$DATA_PATH" = /* ]] || DATA_PATH="$REPO_DIR/$DATA_PATH"
    [[ -d "$TOKENIZER_MODEL" ]] || die "Tokenizer directory does not exist"
    for file in tokenizer.json tokenizer_config.json; do
        [[ -s "$TOKENIZER_MODEL/$file" ]] || die "Missing tokenizer file: $TOKENIZER_MODEL/$file"
    done
    [[ -f "$DATA_PATH.bin" && -f "$DATA_PATH.idx" ]] || die "Missing DATA_PATH.bin or DATA_PATH.idx"
    if [[ "$DRY_RUN" != 1 ]]; then
        "$PYTHON_BIN" - "$TOKENIZER_MODEL" <<'PY'
import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True, use_fast=True)
# Tokenizer entries and the padded model embedding size need not be equal.
assert 200019 <= len(tok) <= 201088, f"Unexpected GPT-OSS tokenizer size: {len(tok)}"
assert max(tok.get_vocab().values()) < 201088, "Token IDs exceed GPT-OSS embeddings"
assert tok.convert_tokens_to_ids("<|start|>") == 200006, "Wrong tokenizer"
assert tok.eos_token_id == 200002, "Expected GPT-OSS <|return|> EOS token"
PY
    fi
    DATA_ARGS+=(--tokenizer-type HuggingFaceTokenizer --tokenizer-model "$TOKENIZER_MODEL" --data-path "$DATA_PATH")
fi

# 5. GPT-OSS 20B: 24 layers; 32 experts/layer, top-4 routing.
# Biased attention/MLPs/router, learned attention sinks, and alternating 128-token
# sliding/full attention. quick-geglu implements x*sigmoid(1.702*x), with
# GPT-OSS's clamp=7 and linear-branch offset=1.
MODEL_ARGS=(
    --num-layers 24
    --hidden-size 2880
    --ffn-hidden-size 2880
    --num-attention-heads 64
    --group-query-attention
    --num-query-groups 8
    --kv-channels 64
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --quick-geglu
    --glu-linear-offset 1.0
    --activation-func-clamp-value 7.0
    --no-bias-dropout-fusion
    --untie-embeddings-and-output-weights
    --position-embedding-type yarn
    --rotary-base 150000
    --rotary-percent 1.0
    --no-rope-fusion
    --softmax-type learnable
    --window-size 127,0
    --window-attn-skip-freq 2
    --max-position-embeddings 131072
    --seq-length "$SEQ_LENGTH"
    --padded-vocab-size 201088
    --make-vocab-size-divisible-by 128
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --init-method-std 0.02
    --transformer-impl transformer_engine
    --attention-backend auto
)
MOE_ARGS=(
    --num-experts 32
    --moe-ffn-hidden-size 2880
    --moe-layer-freq 1
    --moe-router-topk 4
    --moe-router-score-function softmax
    --moe-router-dtype fp32
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff "${MOE_AUX_LOSS_COEFF:-0.01}"
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --moe-per-layer-logging
)
# Auxiliary balancing strength is a smoke-training choice, not a claim about
# OpenAI's original training recipe. Routing applies softmax after top-k.

# 6. BF16 avoids FP16 loss-scaling overhead. Full activation recomputation
# trades additional calculation for lower memory use. Sequence parallelism
# distributes activation work over TP ranks. Full CPU optimizer offload
# minimizes GPU optimizer storage. Transfer overlap is disabled for this smoke run
# to avoid additional concurrent transfer/buffer pressure.
# At DP=1 the distributed optimizer does not provide additional DP sharding.
PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP"
    --pipeline-model-parallel-size "$PP"
    --expert-model-parallel-size "$EP"
    --expert-tensor-parallel-size "$ETP"
    --sequence-parallel
    --use-distributed-optimizer
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)
TRAINING_ARGS=(
    --bf16
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --train-iters "$TRAIN_ITERS"
    --optimizer adam
    --use-precision-aware-optimizer
    --optimizer-cpu-offload
    --optimizer-offload-fraction "$OPTIMIZER_OFFLOAD_FRACTION"
    --lr "${LR:-1.5e-4}"
    --min-lr "${MIN_LR:-1.5e-5}"
    --lr-decay-style cosine
    --lr-decay-iters "$TRAIN_ITERS"
    --lr-warmup-iters "$WARMUP_ITERS"
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --seed 1234
    --log-interval 1
    --log-throughput
    --eval-interval 100
    --eval-iters 0
    --save-interval "$SAVE_INTERVAL"
    --ckpt-format torch_dist
    --tensorboard-dir "$RUN_DIR/tensorboard"
    --data-cache-path "$RUN_DIR/data-cache"
    --distributed-timeout-minutes 30
    
)

# Omitting --save also avoids Megatron's final checkpoint at training exit.
if [[ "$SAVE_CHECKPOINTS" == 1 ]]; then
    TRAINING_ARGS+=(--save "$RUN_DIR/checkpoints")
fi

echo "Nodes=$NUM_NODES GPUs=$WORLD_SIZE TP=$TP PP=$PP DP=$DP EP=$EP ETP=$ETP"
echo "Microbatches/step=$NUM_MICROBATCHES tokens/step=$((GLOBAL_BATCH_SIZE * SEQ_LENGTH))"
echo "CPU optimizer offload fraction=$OPTIMIZER_OFFLOAD_FRACTION output=$RUN_DIR mock_data=$MOCK_DATA"

# 7. Launch locally. Arrays preserve each path/flag as one argument.
ADAPTER="$RUN_DIR/pretrain_gptoss.py"
COMMAND=(
    "$PYTHON_BIN" -m torch.distributed.run
    --standalone --nnodes 1 --nproc_per_node "$GPUS_PER_NODE" --max_restarts 0
    "$ADAPTER"
    "${MODEL_ARGS[@]}" "${MOE_ARGS[@]}" "${PARALLEL_ARGS[@]}" "${TRAINING_ARGS[@]}" "${DATA_ARGS[@]}"
)
if (( ${#CHECKPOINT_ARGS[@]} )); then COMMAND+=("${CHECKPOINT_ARGS[@]}"); fi
if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY_RUN: no training processes started and no output directory created."
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

# Fail before allocating the model if the runtime or GPU allocation is wrong.
"$PYTHON_BIN" - <<'PY'
import torch
import transformer_engine.pytorch
from importlib.metadata import version
from packaging.version import Version
assert Version(torch.__version__.split("+")[0]) >= Version("2.3"), "CPU offload requires PyTorch >=2.3"
assert Version(version("transformer-engine")) >= Version("2.8.0"), "GPT-OSS attention sinks require Transformer Engine >=2.8.0"
assert torch.cuda.device_count() == 8, "Expected exactly 8 visible GPUs"
assert torch.cuda.is_bf16_supported(), "BF16 support is required"
print("PyTorch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0))
PY
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR" || die "Cannot create a fresh RUN_DIR"
# This branch does not wire YaRN attributes through its standard CLI config.
# The adapter supplies them before pretrain_gpt imports its model builders.
cat > "$ADAPTER" <<'PY_ADAPTER'
"""Supply GPT-OSS YaRN settings and run the repository's normal pretraining entry point."""

import runpy
from pathlib import Path

from megatron.training import argument_utils, arguments

_original_config_from_args = argument_utils.core_transformer_config_from_args


def gptoss_config_from_args(*args, **kwargs):
    """Build the usual training config, adding the GPT-OSS YaRN parameters."""
    config = _original_config_from_args(*args, **kwargs)
    config.yarn_rotary_scaling_factor = 32.0
    config.yarn_original_max_position_embeddings = 4096
    config.yarn_beta_fast = 32.0
    config.yarn_beta_slow = 1.0
    config.yarn_mscale = 1.0
    config.yarn_mscale_all_dim = 0.0
    config.yarn_correction_range_round_to_int = False
    return config


argument_utils.core_transformer_config_from_args = gptoss_config_from_args
arguments.core_transformer_config_from_args = gptoss_config_from_args
runpy.run_path(str(Path.cwd() / "pretrain_gpt.py"), run_name="__main__")
PY_ADAPTER
printf '%q ' "${COMMAND[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
"${COMMAND[@]}" 2>&1 | tee "$RUN_DIR/train.log"
