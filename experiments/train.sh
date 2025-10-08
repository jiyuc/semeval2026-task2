#!/bin/bash
#SBATCH --time=24:00:00 --mail-type=END --mail-user=necva.bolucu@csiro.au
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH -A OD-240335
#SBATCH --job-name task_1
#SBATCH --output=slurm_outputs/%x-%A_%a.out

python task_1.py\
	--input_file "data/subtask1_test.csv"\
	--model_name "../models/gpt-oss-20b" \
	--output_file "prediction/gpt_oss_20b_task_1_pred.csv"
