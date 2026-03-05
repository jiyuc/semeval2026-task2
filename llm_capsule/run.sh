#!/bin/bash
set -e

# ================= Configuration =================
ENV_NAME="semeval2026t2"
ENV_YML="environment.yml"

DATA_DIR="data"
MODELS_DIR="models"
LLM_DIR="llm"

HF_MODEL_ID="google/gemma-3-270m"
LOCAL_MODEL_DIR="${LLM_DIR}/gemma-3-270m"

# ================= Conda Setup =================
#source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^${ENV_NAME}\s"; then
    conda env create -n "$ENV_NAME" -f "$ENV_YML"
fi

conda activate "$ENV_NAME"
pip install --upgrade pip

# ================= Directories =================
mkdir -p "$MODELS_DIR" "$LLM_DIR" "$DATA_DIR"

# ================= Download LLM =================
if [ ! -d "$LOCAL_MODEL_DIR" ]; then
python <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$HF_MODEL_ID",
    local_dir="$LOCAL_MODEL_DIR",
    local_dir_use_symlinks=False
)
EOF
fi

# ================= Subtask 1 =================
python task1_regression.py \
  --model_name "$LOCAL_MODEL_DIR" \
  --train_data_file "$DATA_DIR/train-data/subtask1_train_feelings_context.csv" \
  --test_data_file "$DATA_DIR/test-data/test_subtask1_feelings_context.csv" \
  --output_dir "$MODELS_DIR/gemma3_task1_feelings" \
  --context feelings

# ================= Subtask 2a =================
python task2a_regression.py \
  --model_name "$LOCAL_MODEL_DIR" \
  --train_data_file "$DATA_DIR/train-data/subtask2a_train_feelings_context.csv" \
  --test_data_file "$DATA_DIR/test-data/subtask2a_forecasting_user_marker_feelings_context.csv" \
  --output_dir "$MODELS_DIR/gemma3_task2a_feelings" \
  --context feelings

# ================= Subtask 2b =================
python task2b_regression.py \
  --model_name "$LOCAL_MODEL_DIR" \
  --train_data_file "$DATA_DIR/train-data/subtask2b_train_feelings_context.csv" \
  --test_data_file "$DATA_DIR/test-data/subtask2b_forecasting_user_marker_feelings_context.csv" \
  --output_dir "$MODELS_DIR/gemma3_task2b_feelings" \
  --context feelings
