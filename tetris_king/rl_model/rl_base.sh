#!/bin/bash
#SBATCH --job-name=tetris_sweep
#SBATCH --output=logs/rl_base.out  # %A=JobId, %a=ArrayIdx
#SBATCH --error=logs/rl_base.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6              # Give Python 32 cores for the 32 envs
#SBATCH --mem=32G                      # Plenty of RAM
#SBATCH --partition=gpu                # Change to your L40S partition name
#SBATCH --gpus=l40s                    # Request 1 L40S per worker
#SBATCH --time=12:00:00                # Give it 12 hours

# 1. Load your modules
module load conda/latest

# 2. Activate environment
conda activate /home/xbao_olin_edu/.conda/envs/pytorch-latest

# 3. Create logs directory if missing
mkdir -p logs

# 4. Run the sweeper
# The python script connects to the DB and picks the next available trial
python -m tetris_king.rl_model.rl

echo "Job finished at $(date)"
