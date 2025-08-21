#!/bin/bash
#SBATCH --job-name=img2img-turbo-run
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # one launcher per node
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --output=logs_%j.out
#SBATCH --error=logs_%j.err
#SBATCH --time=24:00:00

# Activate environment
source /home/omjadhav/miniconda3/bin/activate img2img-turbo

# Define and log node and distributed config
nodes_array=($(scontrol show hostnames $SLURM_JOB_NODELIST))
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "===================================="
echo "SLURM Job Information"
echo "===================================="
echo "Job Name: $SLURM_JOB_NAME"
echo "Number of Nodes: $SLURM_NNODES"
echo "GPUs per Node: $SLURM_GPUS_ON_NODE"
echo "Nodes Array: ${nodes_array[@]}"
echo "Head Node: $head_node"
echo "Head Node IP: $head_node_ip"
echo "===================================="

# Distributed env vars
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29900
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_GPUS_ON_NODE))
export RANK=$SLURM_PROCID
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# NCCL config
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=INFO
export NCCL_PROTO=simple
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=0
export NCCL_NET=IB 
export TORCHELASTIC_TIMEOUT=180
export NCCL_DEBUG=WARN
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 # load model offline
export TRANSFORMERS_OFFLINE=1

# HF/Accelerate log levels
export TRANSFORMERS_VERBOSITY=info
export HF_LOG_LEVEL=info

echo "===================================="
echo "Distributed Setup"
echo "===================================="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
echo "NCCL_DEBUG: $NCCL_DEBUG"
echo "===================================="

# Launch job using Accelerate
echo "===================================="
echo "Starting Distributed Training with accelerate"
echo "===================================="

# =============================================================================
# CycleGAN-Turbo Training Script Parameter Explanations
# =============================================================================

# DISTRIBUTED TRAINING SETUP:
# --config_file: Configuration file for distributed training setup
# --main_process_ip: IP address of the main process for distributed communication  
# --main_process_port: Port number for main process communication
# --num_processes: Total number of processes across all machines
# --num_machines: Number of compute nodes allocated by SLURM
# --machine_rank: Current machine's rank in the distributed setup
# --rdzv_backend: Rendezvous backend for process coordination

# MODEL AND DATA:
# --pretrained_model_name_or_path: Base model to start training from
# --dataset_folder: Path to training dataset directory
# --train_img_prep: Data augmentation pipeline for training images
# --val_img_prep: Simple preprocessing for validation images

# TRAINING HYPERPARAMETERS:
# --learning_rate: Step size for gradient-based optimization
# --max_train_steps: Maximum number of weight updates before stopping steps = batches / grad accu steps 
# --max_train_epochs: Maximum number of complete dataset passes
# --train_batch_size: Number of images processed simultaneousl. (so if batch size 4 each gpu will get 4 images so if have 4 gpus then 4x4=16 images at a time)
# --gradient_accumulation_steps: Updates and Syncs weights updation Between GPU after given no of batches (so if batch size 4 and accumulation 8 then weights updation will happen after 8 batches of size 4 means 8*4 = 32 images) 

# OPTIMIZATION AND LOGGING:
# --tracker_project_name: Experiment name for logging and tracking
# --enable_xformers_memory_efficient_attention: Use optimized attention mechanism to save GPU memory
# --validation_steps: Run validation every N training steps

# CYCLEGAN LOSS WEIGHTS:
# --lambda_gan: Weight for adversarial loss in total loss function
# --lambda_idt: Weight for identity preservation loss
# --lambda_cycle: Weight for cycle consistency loss

# --output_dir: Directory to save model checkpoints and results

# =============================================================================
# RUN Commnad:
# =============================================================================

srun accelerate launch \
  --config_file accelerate_config.yaml \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --num_processes $WORLD_SIZE \
  --num_machines $SLURM_NNODES \
  --machine_rank $RANK \
  --rdzv_backend c10d \
  src/train_cyclegan_turbo_renew.py \
  --pretrained_model_name_or_path="stabilityai/sd-turbo" \
  --dataset_folder data/EYE \
  --train_img_prep "resize_286_randomcrop_256x256_hflip" \
  --val_img_prep "resize_256" \
  --learning_rate "1e-5" \
  --max_train_steps 50000 \
  --max_train_epochs 100 \
  --stopping_criterion epochs \
  --train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --tracker_project_name "cyclegan_turbo_optimized" \
  --enable_xformers_memory_efficient_attention \
  --validation_steps 250 \
  --lambda_gan 0.5 \
  --lambda_idt 1 \
  --lambda_cycle 1 \
  --dataloader_num_workers 32 \
  --output_dir /scratch/omjadhav/output/ \
  --lr_scheduler "constant" \
  --lr_warmup_steps 0
  
echo "===================================="
echo "Training Completed"
echo "===================================="

