#!/bin/bash
#PBS -q rt_HF
#PBS -l select=2:mpiprocs=1
#PBS -l walltime=02:00:00
#PBS -N llama33_70b_h200
#PBS -P gai51741
#PBS -j oe
#PBS -V

# Random-initialized Llama 3.3 70B architecture; this does not load Meta weights.
# Tested against the flags on the Megatron-LM-Edgar-abci branch.
# Submit from the repository root after activating your working Megatron Python
# environment and loading ABCI's HPC-X module (module load hpcx/2.20).
# Real data: export TOKENIZER_MODEL=/shared/llama33-tokenizer
#            export DATA_PATH=/shared/llama33_arxiv_text_document
#            qsub examples/llama/edo_train_llama33_70b_h200.sh
# Benchmark: qsub -v MOCK_DATA=1 examples/llama/edo_train_llama33_70b_h200.sh
# 100 optimizer steps is a smoke/throughput run, not a complete pretraining recipe.

set -euo pipefail
die() { echo "ERROR: $*" >&2; exit 1; }

# 1. PBS allocates two full H200 nodes, each with eight GPUs. MPI launches ONE
# torchrun agent per node; that agent launches eight GPU training processes.
: "${PBS_NODEFILE:?Submit with qsub; PBS_NODEFILE must describe the allocation}"
[[ -r "$PBS_NODEFILE" ]] || die "Cannot read PBS_NODEFILE=$PBS_NODEFILE"
REPO_DIR=${REPO_DIR:-${PBS_O_WORKDIR:-$PWD}}
cd "$REPO_DIR"
REPO_DIR=$PWD
[[ -f pretrain_gpt.py ]] || die "Submit from the Megatron-LM repository root"
NUM_NODES=$(awk '!seen[$0]++ {n++} END {print n+0}' "$PBS_NODEFILE")
GPUS_PER_NODE=8
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
MASTER_ADDR=${MASTER_ADDR:-$(awk 'NR == 1 {print; exit}' "$PBS_NODEFILE")}
MASTER_PORT=${MASTER_PORT:-6000}
PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
[[ "$PYTHON_BIN" = /* && -x "$PYTHON_BIN" ]] || die "PYTHON_BIN must be an absolute executable path"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}

# 2. TP=8 keeps tensor collectives within a node; PP=2 places 40 layers on each
# of two nodes. With 16 GPUs, DP=16/(8*2)=1. More pairs of nodes increase DP.
TP=8
PP=2
(( NUM_NODES >= 2 && WORLD_SIZE % (TP * PP) == 0 )) || die "Allocate an even number of full nodes (at least two)"
DP=$((WORLD_SIZE / (TP * PP)))
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
SEQ_LENGTH=${SEQ_LENGTH:-2048}
TRAIN_ITERS=${TRAIN_ITERS:-100}
WARMUP_ITERS=${WARMUP_ITERS:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-100}
EXIT_MINUTES=${EXIT_MINUTES:-110}
MOCK_DATA=${MOCK_DATA:-0}
DRY_RUN=${DRY_RUN:-0}
for name in MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE SEQ_LENGTH TRAIN_ITERS SAVE_INTERVAL EXIT_MINUTES MASTER_PORT; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer"
done
[[ "$WARMUP_ITERS" =~ ^(0|[1-9][0-9]*)$ ]] || die "WARMUP_ITERS must be a nonnegative integer"
[[ "$MOCK_DATA" =~ ^[01]$ && "$DRY_RUN" =~ ^[01]$ ]] || die "MOCK_DATA and DRY_RUN must be 0 or 1"
(( MASTER_PORT <= 65535 )) || die "MASTER_PORT exceeds 65535"
(( SEQ_LENGTH <= 131072 && SEQ_LENGTH % TP == 0 )) || die "SEQ_LENGTH must be divisible by 8 and <=131072"
(( WARMUP_ITERS < TRAIN_ITERS )) || die "WARMUP_ITERS must be less than TRAIN_ITERS"
(( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DP) == 0 )) || die "Global batch must be divisible by micro batch times DP"
NUM_MICROBATCHES=$((GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * DP)))

# 3. All paths must be visible at the same location on every allocated node.
# A new output directory prevents accidental reuse/overwriting of another run.
RUN_DIR=${RUN_DIR:-$REPO_DIR/runs/llama33_70b_h200/${PBS_JOBID:-manual-$(date +%Y%m%d-%H%M%S)}}
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
# distributes activation work over TP ranks. The distributed optimizer shards
# optimizer state across DP replicas (there is no DP sharding benefit at DP=1).
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
    --exit-duration-in-mins "$EXIT_MINUTES"
    --distributed-timeout-minutes 30
)

echo "Nodes=$NUM_NODES GPUs=$WORLD_SIZE TP=$TP PP=$PP DP=$DP"
echo "Microbatches/step=$NUM_MICROBATCHES tokens/step=$((GLOBAL_BATCH_SIZE * SEQ_LENGTH))"
echo "Rendezvous=$MASTER_ADDR:$MASTER_PORT output=$RUN_DIR mock_data=$MOCK_DATA"

# 7. Generate a shared worker script with safely quoted argument arrays. PBS
# runs this submission script only once, so MPI explicitly starts both agents.
# Do not use the PBS spool copy of $0 as the remote worker path.
mkdir -p "$RUN_DIR"
WORKER="$RUN_DIR/launch_node.sh"
{
    printf '#!/bin/bash\nset -euo pipefail\n'
    printf 'cd %q\n' "$REPO_DIR"
    printf 'echo "Launching node rank ${OMPI_COMM_WORLD_RANK:?} on $(hostname)"\n'
    printf '%q -c %q\n' "$PYTHON_BIN" 'import torch; import transformer_engine.pytorch; assert torch.cuda.device_count() == 8, "Expected 8 visible GPUs per node"; assert torch.cuda.is_bf16_supported(), "BF16 is required"; print(torch.__version__)'
    printf 'exec %q -m torch.distributed.run ' "$PYTHON_BIN"
    printf '%q ' --nnodes "$NUM_NODES" --nproc_per_node "$GPUS_PER_NODE" --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" --max_restarts 0
    printf '%s ' '--node_rank "${OMPI_COMM_WORLD_RANK:?}"'
    printf '%q ' "$REPO_DIR/pretrain_gpt.py" "${MODEL_ARGS[@]}" "${PARALLEL_ARGS[@]}" "${TRAINING_ARGS[@]}" "${DATA_ARGS[@]}"
    if (( ${#CHECKPOINT_ARGS[@]} )); then printf '%q ' "${CHECKPOINT_ARGS[@]}"; fi
    printf '\n'
} > "$WORKER"

if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY_RUN: generated $WORKER; no training processes started."
    cat "$WORKER"
    exit 0
fi
command -v mpirun >/dev/null || die "Load ABCI HPC-X (module load hpcx/2.20) before submission"
mpirun -np "$NUM_NODES" --map-by ppr:1:node --bind-to none \
    --hostfile "$PBS_NODEFILE" \
    -x PATH -x LD_LIBRARY_PATH -x OMP_NUM_THREADS -x CUDA_DEVICE_MAX_CONNECTIONS \
    /bin/bash "$WORKER" 2>&1 | tee "$RUN_DIR/train.log"
