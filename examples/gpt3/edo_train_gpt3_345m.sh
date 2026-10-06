#!/bin/bash

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

GPUS_PER_NODE=8
# Change for multinode config
# MASTER_ADDR=localhost
# MASTER_PORT=6000

NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

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


CHECKPOINT_PATH=checkpoints/gpt3_345m/8gpu_mp
TENSORBOARD_LOGS_PATH=tensorboard
VOCAB_FILE=dataset/gpt2-vocab.json
MERGE_FILE=dataset/gpt2-merges.txt
DATA_PATH=dataset/arxiv_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)


GPT_MODEL_ARGS=(
    --num-layers 24 
    --hidden-size 1024 
    --num-attention-heads 16 
    --seq-length 1024 
    --max-position-embeddings 1024 
    --attention-backend auto
)

TRAINING_ARGS=(
    --micro-batch-size 4
    --global-batch-size 32
    --train-iters 200
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)

MODEL_PARALLEL_ARGS=(
    --pipeline-model-parallel-size 2
	--tensor-model-parallel-size 2
    --num-workers 10
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    # --profile 
    # --use-pytorch-profiler
    # --profile-step-start 1
    # --profile-step-end 3
    # --profile-ranks 0 1 2 3 4 5 6 7
    # --pytorch-profiler-collect-callstack
    # --pytorch-profiler-collect-shapes
    # --pytorch-profiler-collect-chakra
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}  
