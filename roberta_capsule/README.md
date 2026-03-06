# RoBERTa Capsule for SemEval-2026 Task 2

This directory contains the implementation of a RoBERTa-based system for SemEval-2026 Task 2. The system addresses three subtasks: valence/arousal regression for individual tweets (Subtask 1), state change forecasting (Subtask 2a), and user disposition change forecasting (Subtask 2b).

### Project Structure

- `run.sh`: Main entry point script that sets up the environment, preprocesses data, trains models, and generates predictions.
- `create_cv_split.py`: Script to create cross-validation splits based on `user_id` to ensure no user overlap between train and validation sets.
- `task1_trainer.py` / `task1_predictor.py`: Training and inference scripts for Subtask 1 (Tweet-level Valence/Arousal).
- `task2a_trainer.py` / `task2a_predictor.py`: Training and inference scripts for Subtask 2a (State Change Forecasting).
- `task2b_trainer.py` / `task2b_predictor.py`: Training and inference scripts for Subtask 2b (Disposition Change Forecasting).
- `data/`: Directory where the raw CSV datasets should be placed.
- `requirements.txt`: Python dependencies.
- `description.pdf`: Detailed system description.

### Prerequisite and Config

1. **Conda Setup**: Ensure the conda activation path is correctly specified for `CONDA` in `run.sh` (e.g., `/home/usr/anaconda3/bin/activate`).
2. **Data Placement**: Ensure the following CSV files are stored in the `roberta_capsule/data/` directory (or the path specified as `DATA_DIR` in `run.sh`):
   - `train_subtask1.csv`
   - `train_subtask2a.csv`
   - `train_subtask2b.csv`
   - `test_subtask1.csv`
   - `subtask2a_forecasting_user_marker.csv`
   - `subtask2b_forecasting_user_marker.csv`

The `run.sh` script will automatically:
- Initialize a conda virtual environment named `semeval2026t2` with `python==3.12.2`.
- Install required packages specified in `requirements.txt`.
- Download the pre-trained model `cardiffnlp/twitter-roberta-base-sentiment-latest` from Hugging Face.

### Usage

To reproduce the results from end-to-end, simply run:
```bash
bash run.sh
```

The script performs the following steps:
1. **Environment Setup**: Creates and activates the conda environment.
2. **Data Splitting**: Runs `create_cv_split.py` to generate 5-fold cross-validation splits in `data/split/`.
3. **Subtask 1**: Finetunes RoBERTa for valence and arousal regression and generates predictions for the test set.
4. **Subtask 2a**: Trains the state change forecasting model using historical tweet features and generates predictions.
5. **Subtask 2b**: Trains the disposition change forecasting model and generates predictions.

**Outputs**:
- Finetuned models are saved in the `model/` directory.
- Prediction CSVs are stored in the `results/` directory.

### Training Details

- **Subtask 1**: Uses a RoBERTa backbone with a regression head. Trained for 4 epochs with a learning rate of 1e-5.
- **Subtask 2a**: Incorporates historical features (valence/arousal) and time differences. Trained for 10 epochs with a learning rate of 1e-3.
- **Subtask 2b**: Uses aggregated user features like mean valence/arousal, autocorrelation, and MSSD (Mean Successive Squared Difference). Trained for 10 epochs with a learning rate of 2e-5.

The entire reproduction process takes less than 1 hour on a system with an NVIDIA RTX 3500 Ada Generation (12GB VRAM).

### System Description
A detailed system description is provided in `description.pdf`.