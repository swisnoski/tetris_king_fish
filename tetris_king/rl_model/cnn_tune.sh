#!/bin/bash
#SBATCH --job-name=tetris_sweep
#SBATCH --output=logs/sweep_%A_%a.out  # %A=JobId, %a=ArrayIdx
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # Give Python 32 cores for the 32 envs
#SBATCH --mem=64G                      # Plenty of RAM
#SBATCH --partition=gpu                # Change to your L40S partition name
#SBATCH --gpus=l40s                    # Request 1 L40S per worker
#SBATCH --time=12:00:00                # Give it 12 hours
#SBATCH --array=1-10                   # LAUNCH 10 WORKERS SIMULTANEOUSLY

# 1. Load your modules
module load conda/latest

# 2. Activate environment
conda activate /home/xbao_olin_edu/.conda/envs/pytorch-latest

# 3. Create logs directory if missing
mkdir -p logs

# 4. Run the sweeper
# The python script connects to the DB and picks the next available trial
python -m tetris_king.rl_model.rl_tune

echo "Job finished at $(date)"
