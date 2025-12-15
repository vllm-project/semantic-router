#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=finetuning_complexity
#SBATCH --account=cse585f25_class
#SBATCH --partition=spgpu,gpu_mig40
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8g
#SBATCH --output=out_reg_%j.txt

source /scratch/cse585f25_class_root/cse585f25_class/anikrish/fine-tune-factual/.venv/bin/activate

python finetuning_regression.py
