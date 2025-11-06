#!/bin/bash
#SBATCH --job-name=finetune-task1  # Job name
#SBATCH --output=logs/%x_%j.out    # Standard output
#SBATCH --error=logs/%x_%j.err     # Standard error
#SBATCH --time=08:00:00            # Max runtime (HH:MM:SS)
#SBATCH --cpus-per-task=4          # Number of CPU cores
#SBATCH --mem=50G                  # Memory
#SBATCH --gres=gpu:1               # Uncomment if GPU is needed
#SBATCH -A OD-240335



MODEL_PATH="../models/gemma-3-270m"
OUTPUT_BASE="models/gemma3_270m_task1_regression"
SCRIPT="task1_gemma_regression.py"

for fold in {1..5}
do
    TRAIN_FILE="split/feeling_dataset/feeling_constraint/subtask1_train_cv${fold}.csv"
    TEST_FILE="split/feeling_dataset/feeling_constraint/subtask1_test_cv${fold}.csv"
    OUTPUT_DIR="${OUTPUT_BASE}_fold${fold}"

    echo "Running fold $fold..."
    python $SCRIPT \
        --model_name $MODEL_PATH \
        --train_data_file $TRAIN_FILE \
        --test_data_file $TEST_FILE \
        --output_dir $OUTPUT_DIR \
		--context "both"
done
