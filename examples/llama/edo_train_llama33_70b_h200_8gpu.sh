#!/bin/bash
# Run from the repository root in your active Megatron environment:
#   bash examples/llama/edo_train_llama33_70b_h200_8gpu.sh
# Synthetic data: MOCK_DATA=1 bash examples/llama/edo_train_llama33_70b_h200_8gpu.sh

python --version
python -c "import torch; print(torch.__version__)"

# Runs the "175B" parameter model
# Change for multinode config
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1|head -n 1)
MASTER_PORT=6000
# export MASTER_PORT=$((10000 + (13450 % 50000)))
    echo "MASTER_ADDR: ${MASTER_ADDR}"
    echo "MASTER_PORT: ${MASTER_PORT}"
export CUDA_DEVICE_MAX_CONNECTIONS=1


# Removing Checkpoint
echo removing checkpoints/....
rm -rf checkpoints/*

# Setting up OMP_THREADS
export OMP_NUM_THREADS=4

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
[[ -f pretrain_gpt.py ]] || die "Run from the Megatron-LM repository root"
NUM_NODES=1
GPUS_PER_NODE=8
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
[[ "$PYTHON_BIN" = /* && -x "$PYTHON_BIN" ]] || die "PYTHON_BIN must be an absolute executable path"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}

# 2. Each of the 80 layers is split across all eight GPUs: TP=8, PP=1, DP=1.
# CPU optimizer offload creates memory headroom for full training.
TP=8
PP=1
DP=$((WORLD_SIZE / (TP * PP)))
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
SEQ_LENGTH=${SEQ_LENGTH:-2048}
TRAIN_ITERS=${TRAIN_ITERS:-100}
WARMUP_ITERS=${WARMUP_ITERS:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-100}
MOCK_DATA=${MOCK_DATA:-0}
DRY_RUN=${DRY_RUN:-0}
for name in MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE SEQ_LENGTH TRAIN_ITERS SAVE_INTERVAL OMP_NUM_THREADS; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer"
done
[[ "$WARMUP_ITERS" =~ ^(0|[1-9][0-9]*)$ ]] || die "WARMUP_ITERS must be a nonnegative integer"
[[ "$MOCK_DATA" =~ ^[01]$ && "$DRY_RUN" =~ ^[01]$ ]] || die "MOCK_DATA and DRY_RUN must be 0 or 1"
(( SEQ_LENGTH <= 131072 && SEQ_LENGTH % TP == 0 )) || die "SEQ_LENGTH must be divisible by 8 and <=131072"
(( WARMUP_ITERS < TRAIN_ITERS )) || die "WARMUP_ITERS must be less than TRAIN_ITERS"
(( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DP) == 0 )) || die "Global batch must be divisible by micro batch times DP"
NUM_MICROBATCHES=$((GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * DP)))

# 3. Use persistent storage for datasets, logs, and the large checkpoints.
# A new output directory prevents accidental reuse/overwriting of another run.
RUN_DIR=${RUN_DIR:-$REPO_DIR/runs/llama33_70b_h200_8gpu/interactive-$(date +%Y%m%d-%H%M%S)-$}
[[ "$RUN_DIR" = /* ]] || RUN_DIR="$REPO_DIR/$RUN_DIR"
[[ ! -e "$RUN_DIR" ]] || die "RUN_DIR already exists; choose a fresh directory"
RESUME_FROM=${RESUME_FROM:-}
CHECKPOINT_ARGS=()
if [[ -n "$RESUME_FROM" ]]; then
    [[ "$RESUME_FROM" = /* ]] || RESUME_FROM="$REPO_DIR/$RESUME_FROM"
    [[ -f "$RESUME_FROM/latest_checkpointed_iteration.txt" ]] || die "RESUME_FROM is not a Megatron checkpoint directory"
    CHECKPOINT_ARGS+=(--load "$RESUME_FROM" --exit-on-missing-checkpoint)
fi

# 4. Llama's token IDs are incompatible with the GPT-2 vocabulary used by the
# original GPT-3 script. Re-tokenize the original text with THIS tokenizer.
DATA_ARGS=(--split 949,50,1 --num-workers 4 --no-create-attention-mask-in-dataloader)
if [[ "$MOCK_DATA" == 1 ]]; then
    DATA_ARGS+=(--mock-data --tokenizer-type NullTokenizer --vocab-size 128256)
else
    : "${TOKENIZER_MODEL:?Set a shared local directory containing the Llama 3.3 HF tokenizer files}"
    : "${DATA_PATH:?Set the Llama-tokenized Megatron dataset prefix, without .bin or .idx}"
    [[ "$TOKENIZER_MODEL" = /* ]] || TOKENIZER_MODEL="$REPO_DIR/$TOKENIZER_MODEL"
    [[ "$DATA_PATH" = /* ]] || DATA_PATH="$REPO_DIR/$DATA_PATH"
    [[ -d "$TOKENIZER_MODEL" ]] || die "Tokenizer directory does not exist"
    [[ -f "$DATA_PATH.bin" && -f "$DATA_PATH.idx" ]] || die "Missing DATA_PATH.bin or DATA_PATH.idx"
    if [[ "$DRY_RUN" != 1 ]]; then
        "$PYTHON_BIN" - "$TOKENIZER_MODEL" <<'PY'
import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
assert len(tok) == 128256, f"Expected Llama vocabulary of 128256, got {len(tok)}"
PY
    fi
    DATA_ARGS+=(--tokenizer-type HuggingFaceTokenizer --tokenizer-model "$TOKENIZER_MODEL" --data-path "$DATA_PATH")
fi

# 5. Llama 3.3 70B architecture: 80 decoder layers, GQA with 64 query heads and
# 8 KV heads, SwiGLU, RMSNorm, separate input/output embeddings, and scaled RoPE.
# The context configuration matches Llama 3.3, but this short run only trains
# SEQ_LENGTH tokens at a time; it does not establish 128K-context capability.
MODEL_ARGS=(
    --num-layers 80
    --hidden-size 8192
    --ffn-hidden-size 28672
    --num-attention-heads 64
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --swiglu
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --rotary-base 500000
    --rotary-percent 1.0
    --use-rope-scaling
    --rope-scaling-factor 8
    --max-position-embeddings 131072
    --seq-length "$SEQ_LENGTH"
    --make-vocab-size-divisible-by 1
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --init-method-std 0.02
    --transformer-impl transformer_engine
    --attention-backend auto
)

# 6. BF16 avoids FP16 loss-scaling overhead. Full activation recomputation
# trades additional calculation for lower memory use. Sequence parallelism
# distributes activation work over TP ranks. Full CPU optimizer offload moves
# optimizer updates/states to host RAM; transfers overlap with CPU updates.
# At DP=1 the distributed optimizer does not provide additional DP sharding.
PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP"
    --pipeline-model-parallel-size "$PP"
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
    --optimizer-offload-fraction 1.0
    --overlap-cpu-optimizer-d2h-h2d
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
    --eval-iters 5
    --save-interval "$SAVE_INTERVAL"
    --ckpt-format torch_dist
    --save "$RUN_DIR/checkpoints"
    --tensorboard-dir "$RUN_DIR/tensorboard"
    --data-cache-path "$RUN_DIR/data-cache"
    --distributed-timeout-minutes 30
    
)

echo "Nodes=$NUM_NODES GPUs=$WORLD_SIZE TP=$TP PP=$PP DP=$DP"
echo "Microbatches/step=$NUM_MICROBATCHES tokens/step=$((GLOBAL_BATCH_SIZE * SEQ_LENGTH))"
echo "CPU optimizer offload=100% output=$RUN_DIR mock_data=$MOCK_DATA"

# 7. Launch locally. Arrays preserve each path/flag as one argument.
COMMAND=(
    "$PYTHON_BIN" -m torch.distributed.run
    --standalone --nnodes 1 --nproc_per_node "$GPUS_PER_NODE" --max_restarts 0
    "$REPO_DIR/pretrain_gpt.py"
    "${MODEL_ARGS[@]}" "${PARALLEL_ARGS[@]}" "${TRAINING_ARGS[@]}" "${DATA_ARGS[@]}"
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
from packaging.version import Version
assert Version(torch.__version__.split("+")[0]) >= Version("2.3"), "CPU offload requires PyTorch >=2.3"
assert torch.cuda.device_count() == 8, "Expected exactly 8 visible GPUs"
assert torch.cuda.is_bf16_supported(), "BF16 support is required"
print("PyTorch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0))
PY
mkdir -p "$RUN_DIR"
printf '%q ' "${COMMAND[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
"${COMMAND[@]}" 2>&1 | tee "$RUN_DIR/train.log"
