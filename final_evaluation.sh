#!/bin/bash
#SBATCH --job-name=final_eval
#SBATCH --account=cse585f25_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=00:10:00   
#SBATCH --mem-per-cpu=8g
#SBATCH --output=final_eval_out.txt
source /scratch/cse585f25_class_root/cse585f25_class/anikrish/fine-tune-factual/.venv/bin/activate

python final_evaluation.py