#!/bin/bash

#################################################
#SBATCH -J farsight
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-task 8
#SBATCH --gpus-per-node 8
#SBATCH -o /home/user23001/workspace/logs/slurm/%x_%j.log
#SBATCH -e /home/user23001/workspace/logs/slurm/%x_%j.log
#SBATCH -p 023-partition
#SBATCH --wait-all-nodes 1
export GPUS_PER_NODE=8
set -euxo pipefail

## 各タスクがマスターノードを見つけられるように環境変数を設定
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 1024-65535 -n 1)

## InfiniBandを使う場合
export MELLANOX_VISIBLE_DEVICES=all
export UCX_TLS=rc,cuda,sm,self

## TCP/IPでプロセス間通信を行う場合
# export UCX_TLS=tcp,cuda,sm,self

## IntelMPIを使ってInfiniBandでプロセス間通信を行う場合
export FI_LOG_LEVEL=1
export FI_PROVIDER=mlx

## NCCLライブラリを使う場合に切り分けのためにログを表示させる
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

## Default settings
DATE=$(date +%Y-%m-%d)
YYMMDD=$(date +%y%m%d)

export HF_TOKEN=$HF_TOKEN
export HF_HOME=$HF_HOME

#================================================#
## User's settings
export WANDB_PROJECT=SBNasubi
export WANDB_NAME="sft_${SLURM_JOB_ID}"
#================================================#

## Path settings
: "${RESULT_PATH:=/lustre/k_ishikawa/results}" # 結果を保存するディレクトリ
: "${IMAGE:=}" # 実行するsqshイメージの指定
: "${WORK_DIR:=/$HOME}" # 作業ディレクトリの指定

: "${RESULT_MOUNT:=$RESULT_PATH:$RESULT_PATH}" # 結果マウントポイントの指定
: "${WORK_MOUNT:=$WORK_DIR:$WORK_DIR}" # 作業ディレクトリマウントポイントの指定

declare -a CONTAINER_OPTIONS=(
    --container-image=/lustre/k_ishikawa/sqshes/z4pymjpdqjsj+spiralai2+sptrain+0.5.0.sqsh
    --container-mounts=/lustre:/lustre,/home/user23001:/home/user23001
)

## Execute
echo "Starting training..."
srun -l "${CONTAINER_OPTIONS[@]}" \
    accelerate launch \
        --num_processes 4 \
        --num_machines $SLURM_NNODES \
        --rdzv_backend c10d \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        --multi_gpu \
        $HOME/workspace/multi-task/training/farsight/run_branch.py
echo "Training completed."
