#!/bin/bash


# Runs the "345M" parameter model
# Change for multinode config
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1|head -n 1)
export MASTER_PORT=$((10000 + (13450 % 50000)))
    echo "MASTER_ADDR: ${MASTER_ADDR}"
    echo "MASTER_PORT: ${MASTER_PORT}"

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
# MASTER_ADDR=localhost
# MASTER_PORT=6000

# Removing Checkpoint
echo removing checkpoints/gpt2_345m/8gpu_mp...
rm -rf checkpoints/gpt2_345m/8gpu_mp

# Setting up OMP_THREADS
export OMP_NUM_THREADS=4

# Removing tensor-records
echo removing tensor-records...
rm -rf tensorboard/*

CHECKPOINT_PATH=checkpoints/gpt2_345m/8gpu_mp
VOCAB_FILE=dataset/gpt2-vocab.json
MERGE_FILE=dataset/gpt2-merges.txt
DATA_PATH=dataset/arxiv_text_document

DISTRIBUTED_ARGS="
  --nproc_per_node ${GPUS_PER_NODE}
  --nnodes ${NNODES}
  --node_rank ${NODE_RANK}
  --master_addr ${MASTER_ADDR}
  --master_port ${MASTER_PORT}
"

# For logging to TensorBoard, add the following arguments to the command line:
  # --tensorboard-dir tensorboard/
  # --log-memory-to-tensorboard
  # --log-optimizer-states-to-tensorboard
  # --log-timers-to-tensorboard
  # --log-validation-ppl-to-tensorboard
  # --log-world-size-to-tensorboard

GPT_ARGS="
  --tensorboard-dir tensorboard/
  --profile pt-full
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --num-experts 1
  --expert-interval 1
  --num-workers 10
  --num-layers 24
  --hidden-size 1024
  --num-attention-heads 16
  --seq-length 1024
  --max-position-embeddings 1024
  --micro-batch-size 4
  --global-batch-size 32
  --lr 0.00015
  --train-iters 100
  --lr-decay-iters 320000
  --lr-decay-style cosine
  --min-lr 1.0e-5
  --weight-decay 1e-2
  --lr-warmup-fraction .01
  --clip-grad 1.0
  --fp16
"

DATA_ARGS="
  --data-path ${DATA_PATH}
  --vocab-file ${VOCAB_FILE}
  --merge-file ${MERGE_FILE}
  --data-impl mmap
  --split 949,50,1
"

OUTPUT_ARGS="
  --log-interval 10
  --save-interval 10000
  --eval-interval 1000
  --eval-iters 10
"

  

torchrun ${DISTRIBUTED_ARGS} pretrain_gpt.py \
  ${GPT_ARGS} \
  ${DATA_ARGS} \
  ${OUTPUT_ARGS} \
  --distributed-backend nccl 
  
  
  # --save ${CHECKPOINT_PATH} \
  # --load ${CHECKPOINT_PATH}