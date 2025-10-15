#python3 task1_trainer.py \
#  --train data/SemEval2026/data/split/subtask1_train.csv \
#  --validation data/SemEval2026/data/split/subtask1_test.csv \
#  --model cardiffnlp/twitter-xlm-roberta-base \
#  --output_dir ./results \
#  --save_dir ./semeval2026_subtask1_xlm_valence_reg_base \
#  --epochs 5 \
#  --train_bs 16 \
#  --eval_bs 32 \
#  --learning_rate 2e-5 \
#  --label "valence" \
#  --scale_min -2 \
#  --scale_max 2



python3 diff_trainer.py \
  --train data/SemEval2026/data/split/subtask2a_train.csv \
  --validation data/SemEval2026/data/split/subtask2a_test.csv \
  --model cardiffnlp/twitter-xlm-roberta-base \
  --output_dir ./results \
  --save_dir ./semeval2026_subtask2a_xlm_arousal_reg_base \
  --epochs 5 \
  --train_bs 16 \
  --eval_bs 32 \
  --learning_rate 2e-5 \
  --label "state_change_arousal" \
  --feature "arousal"