#!/bin/bash
#SBATCH --job-name=tetris_tune
#SBATCH --output=tetris_tune_%j.out
#SBATCH --error=tetris_tune_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=l40s
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=xbao@olin.edu

# Load modules (adjust based on your HPC's available modules)
module conda/latest

# Activate your conda environment
conda activate pytorch_latest

# Navigate to your project directory
cd ~/comprobo25/tetris_king_fish

# Run the tuning script
python -m tetris_king.rl_model.tune_optuna

echo "Job finished at $(date)"