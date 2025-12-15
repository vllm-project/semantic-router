#!/bin/bash
#SBATCH --job-name=eval_matrix
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=00:10:00   
#SBATCH --mem-per-cpu=8g
#SBATCH --output=eval_out.txt
source /scratch/cse585f25_class_root/cse585f25_class/anikrish/fine-tune-factual/.venv/bin/activate

# Run the confusion matrix python script
python matrix.py