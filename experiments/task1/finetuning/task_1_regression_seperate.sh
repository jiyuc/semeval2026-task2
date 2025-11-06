#!/bin/bash
#SBATCH --job-name=s-finetune-task1  # Job name
#SBATCH --output=logs/%x_%j.out    # Standard output
#SBATCH --error=logs/%x_%j.err     # Standard error
#SBATCH --time=08:00:00            # Max runtime (HH:MM:SS)
#SBATCH --cpus-per-task=4          # Number of CPU cores
#SBATCH --mem=50G                  # Memory
#SBATCH --gres=gpu:1               # Uncomment if GPU is needed
#SBATCH -A OD-240335



MODEL_PATH="../models/gemma-3-1b-it"
OUTPUT_BASE="models/gemma_3_1b_it_task1_regression_seperate"
SCRIPT="task1_gemma_regression_seperate.py"

for fold in {1..5}
do
    TRAIN_FILE="split/subtask1_train_cv${fold}.csv"
    TEST_FILE="split/subtask1_test_cv${fold}.csv"
    OUTPUT_DIR="${OUTPUT_BASE}_fold${fold}"

    echo "Running fold $fold..."
    python $SCRIPT \
        --model_name $MODEL_PATH \
        --train_data_file $TRAIN_FILE \
        --test_data_file $TEST_FILE \
        --output_dir $OUTPUT_DIR \
		    --context "both"
done
