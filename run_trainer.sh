#for i in {1..5}
#do
#  python3 task1_trainer.py \
#    --train data/SemEval2026/data/split/subtask1_train_cv${i}.csv \
#    --validation data/SemEval2026/data/split/subtask1_test_cv${i}.csv \
#    --model cardiffnlp/twitter-xlm-roberta-base \
#    --output_dir ./results \
#    --save_dir ./semeval2026_subtask1_xlm_arousal_reg_base \
#    --epochs 5 \
#    --max_length 256 \
#    --train_bs 16 \
#    --eval_bs 32 \
#    --learning_rate 2e-5 \
#    --label arousal \
#    --report_to none \
#    --scale_min 0 \
#    --scale_max 2
#done
#
#
### inference
#python3 task1_inference.py \
#  --csv data/subtask1_eval.csv \
#  --model_dir models/xlm_valence_reg_model \
#  --target valence \
#  --text_col text \
#  --batch_size 32 \
#  --max_length 256 \
#  --output_csv results/subtask1_eval.valence.predicted.csv
#
## Example: predict arousal (same CSV/schema), writing alongside the input (no --output_csv)
#python3 task1_inference.py \
#  --csv data/subtask1_eval.csv \
#  --model_dir models/semeval2026_subtask1_xlm_arousal_reg_base \
#  --target arousal \
#  --text_col text



## Task 2a
### Finetune
for i in {1..5}
do
  TRAIN_CSV="data/SemEval2026/data/split/subtask2a_train_cv${i}.csv"
  TEST_CSV="data/SemEval2026/data/split/subtask2a_test_cv${i}.csv"
  BASE_MODEL="cardiffnlp/twitter-xlm-roberta-base"

  python3 joint_state_exponential_decay.py \
    --train ${TRAIN_CSV} \
    --validation ${TEST_CSV} \
    --model ${BASE_MODEL} \
    --output_dir ./results \
    --save_dir ./joint_decay_semeval2026_subtask2a_xlm_valence_reg_base \
    --epochs 5 \
    --train_bs 16 \
    --eval_bs 32 \
    --learning_rate 2e-5 \
    --label "state_change_valence" \
    --feature "valence" \
    --report_to none
done

for i in {1..5}
do
  TRAIN_CSV="data/SemEval2026/data/split/subtask2a_train_cv${i}.csv"
  TEST_CSV="data/SemEval2026/data/split/subtask2a_test_cv${i}.csv"
  BASE_MODEL="cardiffnlp/twitter-xlm-roberta-base"

  python3 joint_state_exponential_decay.py \
    --train ${TRAIN_CSV} \
    --validation ${TEST_CSV} \
    --model ${BASE_MODEL} \
    --output_dir ./results \
    --save_dir ./joint_decay_semeval2026_subtask2a_xlm_arousal_reg_base \
    --epochs 5 \
    --train_bs 16 \
    --eval_bs 32 \
    --learning_rate 2e-5 \
    --label "state_change_arousal" \
    --feature "arousal" \
    --report_to none
done


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