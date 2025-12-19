#!/bin/bash
#SBATCH --account=OD-235037
#SBATCH --job-name=num
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

#for i in {1..5}
#do
#  python3 task1_trainer.py \
#    --train data/split/subtask1_train_cv${i}.csv \
#    --validation data/split/subtask1_test_cv${i}.csv \
#    --model cardiffnlp/twitter-roberta-base-emotion-latest \
#    --output_dir ./results \
#    --save_dir ./model/subtask1_arousal_base_cv${i} \
#    --epochs 3 \
#    --max_length 256 \
#    --train_bs 16 \
#    --eval_bs 32 \
#    --dropout_rate 0.2 \
#    --learning_rate 1e-5 \
#    --label arousal \
#    --report_to wandb \
#    --run_name twitter_roberta_emotion_base_cnt_arousal_2e-1_cv${i} \
#    --continual 1
#done
#--calibration_data data/hallmark.csv
#cardiffnlp/twitter-xlm-roberta-base \

#
#
### inference
#python3 task1_inference.py \
#  --csv data/subtask1_eval.csv \
#  --model_dir models/semeval2026_subtask1_xlm_valence_reg_base \
#  --target valence \
#  --text_col text \
#  --batch_size 32 \
#  --max_length 256 \
#  --output_csv results/subtask1_eval.valence.predicted.csv
#
## Example: predict arousal (same CSV/schema), writing alongside the input (no --output_csv)
#python3 task1_inference.py \
#  --csv data/hallmark.csv \
#  --model_dir semeval2026_subtask1_xlm_valence_reg_base_cv1 \
#  --target pred_valence \
#  --text_col text \
#  --output_csv data/hallmark.csv

#data/SemEval2026/data/train_subtask1.csv \


## Task 2a
### Finetune

for COL in valence arousal;
do
  for i in {1..4}
  do
    TRAIN_CSV="data/split/subtask2a_train_cv${i}.csv"
    TEST_CSV="data/split/subtask2a_test_cv${i}.csv"
    BASE_MODEL="cardiffnlp/twitter-xlm-roberta-base"

    python3 task2a_trainer_num.py \
        --run_name xlm_roberta_base_num_latest_${COL}_cv${i} \
        --train ${TRAIN_CSV} \
        --validation ${TEST_CSV} \
        --model ${BASE_MODEL} \
        --output_dir ./results \
        --epochs 3 \
        --train_bs 16 \
        --eval_bs 32 \
        --learning_rate 2e-5 \
        --label state_change_${COL} \
        --feature ${COL} \
        --report_to "wandb"
  done
done

#
#for i in {1..5}
#do
#  TRAIN_CSV="data/split/subtask2a_train_cv${i}.csv"
#  TEST_CSV="data/split/subtask2a_test_cv${i}.csv"
#  BASE_MODEL="cardiffnlp/twitter-xlm-roberta-base"
#
#  python3 joint_state_exponential_decay.py \
#    --run_name subtask2a_xlm_roberta_base_ed_valence_cv${i} \
#    --train ${TRAIN_CSV} \
#    --validation ${TEST_CSV} \
#    --model ${BASE_MODEL} \
#    --output_dir ./results \
#    --epochs 5 \
#    --train_bs 16 \
#    --eval_bs 32 \
#    --learning_rate 2e-5 \
#    --label "state_change_valence" \
#    --feature "valence" \
#    --report_to "none"
#done

#for i in {1..5}
#do
#  TRAIN_CSV="data/split/subtask2a_train_cv${i}.csv"
#  TEST_CSV="data/split/subtask2a_test_cv${i}.csv"
#  BASE_MODEL="cardiffnlp/twitter-xlm-roberta-base"
#
#  python3 joint_inferred_metadata_state_exponential_decay.py \
#    --run_name subtask2a_xlm_roberta_base_ed_imeta_valence_cv${i} \
#    --train ${TRAIN_CSV} \
#    --validation ${TEST_CSV} \
#    --model ${BASE_MODEL} \
#    --output_dir ./results \
#    --epochs 5 \
#    --train_bs 16 \
#    --eval_bs 32 \
#    --learning_rate 2e-5 \
#    --label "state_change_valence" \
#    --feature "valence" \
#    --report_to "wandb"
#done

#for i in {1..5}
#do
#  TRAIN_CSV="data/SemEval2026/data/split/subtask2a_train_cv${i}.csv"
#  TEST_CSV="data/SemEval2026/data/split/subtask2a_test_cv${i}.csv"
#  BASE_MODEL="cardiffnlp/twitter-xlm-roberta-base"
#
#  python3 joint_state_exponential_decay.py \
#    --train ${TRAIN_CSV} \
#    --validation ${TEST_CSV} \
#    --model ${BASE_MODEL} \
#    --output_dir ./results \
#    --save_dir ./joint_decay_semeval2026_subtask2a_xlm_arousal_reg_base \
#    --epochs 5 \
#    --train_bs 16 \
#    --eval_bs 32 \
#    --learning_rate 2e-5 \
#    --label "state_change_arousal" \
#    --feature "arousal" \
#    --report_to none
#done


### Inference
#CSV_IN="data/SemEval2026/data/split/subtask2a_test.state_change_valence.predicted.csv"
#BATCH=32
#MAXLEN=512
#
## Example 1: predict state_change_valence using valence as the pairing feature
#python3 task2a_inference.py \
#  --csv "${CSV_IN}" \
#  --model_dir "./semeval2026_subtask2a_xlm_valence_reg_base" \
#  --target state_change_valence \
#  --feature valence \
#  --batch_size "${BATCH}" \
#  --max_length "${MAXLEN}" \
#  --output_csv "${CSV_IN%.csv}.state_change.predicted.csv"
#
## Example 2: predict state_change_arousal using arousal as the pairing feature
#python3 task2a_inference.py \
#  --csv "${CSV_IN%.csv}.state_change.predicted.csv" \
#  --model_dir "./semeval2026_subtask2a_xlm_arousal_reg_base" \
#  --target state_change_arousal \
#  --feature arousal \
#  --batch_size "${BATCH}" \
#  --max_length "${MAXLEN}" \
#  --output_csv "${CSV_IN%.csv}.state_change.predicted.csv"