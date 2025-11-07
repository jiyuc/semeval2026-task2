#!/bin/bash
#SBATCH --job-name=finetune-task1  # Job name
#SBATCH --output=logs/%x_%j.out    # Standard output
#SBATCH --error=logs/%x_%j.err     # Standard error
#SBATCH --time=08:00:00  --mail-type=END --mail-user=necva.bolucu@csiro.a          # Max runtime (HH:MM:SS)
#SBATCH --cpus-per-task=4          # Number of CPU cores
#SBATCH --mem=50G                  # Memory
#SBATCH --gres=gpu:1               # Uncomment if GPU is needed
#SBATCH -A OD-240335



MODEL_PATH="../models/gemma-3-270m"
OUTPUT_BASE="models/gemma3_270m_task2a_regression"
SCRIPT="task2a_regression.py"

for fold in {1..5}
do
    TRAIN_FILE="split/feeling_dataset/feeling_constraint/subtask2a_train_cv${fold}.csv"
    TEST_FILE="split/feeling_dataset/feeling_constraint/subtask2a_test_cv${fold}.csv"

    for context in text feelings both
    do
        OUTPUT_DIR="${OUTPUT_BASE}_fold${fold}_${context}"

        echo "Running fold $fold with context: $context ..."
        python $SCRIPT \
            --model_name $MODEL_PATH \
            --train_data_file $TRAIN_FILE \
            --test_data_file $TEST_FILE \
            --output_dir $OUTPUT_DIR \
            --context "$context"
    done
done
