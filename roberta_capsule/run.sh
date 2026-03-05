#!/bin/bash


# -------- configuration --------
ENV_NAME="semeval2026t2"
CONDA="/home/usr/anaconda3/bin/activate"  # modify here and run this script in bash
PYTHON_VERSION="3.12.2"
REQ_FILE="requirements.txt"
DATA_DIR=./data/


# Initialize conda for non-interactive shell
source $CONDA semeval2026t2
#source "$(conda info --base)/etc/profile.d/conda.sh"

# Create env if it does not exist
if ! conda env list | grep -q "^$ENV_NAME"; then
    echo "Creating new conda environment: $ENV_NAME"
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

# Activate env
echo "Activating conda environment: $ENV_NAME"
conda activate "$ENV_NAME"

# Install dependencies
if [[ -f "$REQ_FILE" ]]; then
    pip3 install --upgrade pip
    pip3 install -r "$REQ_FILE"
fi

# create underlying directories

mkdir model
mkdir results
mkdir ${DATA_DIR}split

# create cross-validation split
# specify the file dir of three subtasks csv file to create cross-validation subsets
python3 create_cv_split.py --source_data_dir $DATA_DIR


# -------- Finetuning and Inference --------
# -------- Subtask 1 --------
# finetuning
# execute the following bash scripts for finetuning a roberta-sentiment analyzer for valence and arousal regression
for COL in valence arousal;
do
  python3 task1_trainer.py \
      --train ${DATA_DIR}train_subtask1.csv \
      --model cardiffnlp/twitter-roberta-base-sentiment-latest \
      --output_dir ./results \
      --save_dir ./model/twitter-roberta-base-${COL}-latest \
      --epochs 4 \
      --max_length 512 \
      --train_bs 16 \
      --dropout_rate 0.2 \
      --learning_rate 1e-5 \
      --label arousal \
      --report_to "none" \
      --run_name twitter-roberta-base-${COL}-latest
done



# task 1 inference
python3 task1_predictor.py \
    --input_csv ${DATA_DIR}test_subtask1.csv \
    --output_csv ./results/pred_subtask1.csv \
    --model_dir ./model/ \
    --pred_col pred_



# -------- Subtask 2a --------
# finetuning
for COL in valence arousal;
do
python3 task2a_trainer.py \
      --train ${DATA_DIR}split/subtask2a_train_cv3.csv \
      --validation ${DATA_DIR}split/subtask2a_test_cv3.csv \
      --model cardiffnlp/twitter-roberta-base-sentiment-latest \
      --output_dir ./results \
      --save_dir ./model/twitter-roberta-base-state_change_${COL}-latest \
      --epochs 10 \
      --max_length 512 \
      --train_bs 16 \
      --eval_bs 32 \
      --learning_rate 1e-3 \
      --label state_change_${COL} \
      --feature ${COL} \
      --at_N 5 \
      --report_to "none" \
      --run_name final
done


# subtask2a inference
python3 task2a_predictor.py \
  --input_csv ${DATA_DIR}subtask2a_forecasting_user_marker.csv \
  --output_csv ./results/pred_subtask2a.csv \
  --at_N 5 \
  --model_dir ./model/twitter-roberta-base-state_change_{}-latest \
  --base_model cardiffnlp/twitter-roberta-base-sentiment-latest


# -------- Subtask 2b --------
# finetuning
for COL in valence arousal;
do
    python3 task2b_trainer.py \
          --train ${DATA_DIR}split/subtask2b_train_cv3.csv \
          --validation ${DATA_DIR}split/subtask2b_test_cv3.csv \
          --save_dir ./model/twitter-roberta-base-disposition_change_${COL}-latest \
          --model cardiffnlp/twitter-roberta-base-sentiment-latest \
          --output_dir ./results \
          --epochs 10 \
          --max_length 512 \
          --train_bs 16 \
          --eval_bs 32 \
          --learning_rate 2e-5 \
          --label ${COL} \
          --at_N 15 \
          --report_to "none" \
          --run_name final_submission

done


# 2b inference
python3 task2b_predictor.py \
  --input_csv ~/Documents/semeval2026-t2/data/subtask2b_forecasting_user_marker.csv \
  --output_csv ./results/pred_subtask2b.csv \
  --at_N 15 \
  --model_dir ./model/twitter-roberta-base-disposition_change_{}-latest \
  --base_model cardiffnlp/twitter-roberta-base-sentiment-latest





