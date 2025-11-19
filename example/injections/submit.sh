#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=10G
#SBATCH --output="log.out"
#SBATCH --job-name="test_injection"

now=$(date)
echo "$now"

# Loading modules
# module load 2024
# module load Python/3.10.4-GCCcore-11.3.0
conda activate /home/mdrent/miniconda3/envs/Jim

# Display GPU name
nvidia-smi --query-gpu=name --format=csv,noheader

# Run the script
python injection_recovery.py \
    --n-loop-training 3 \
    --n-loop-production 3 \
    --n-local-steps 10 \
    --n-global-steps 10 \
    --waveform-approximant TaylorF2QM_taper \
    --use-QM True \
    --use-f-stop True \