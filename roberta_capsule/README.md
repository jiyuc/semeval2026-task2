### Prerequisite and Config

Ensure the conda path is specified to `CONDA` in `run.sh`
Ensure the following csv files are stored in the same directory specified to `DATA_DIR` in the script `run.sh`
- train_subtask1.csv
- train_subtask2a.csv
- train_subtask2b.csv
- test_subtask1.csv
- subtask2a_forecasting_user_marker.csv
- subtask2b_forecasting_user_marker.csv

The script will automatically initialize a conda virtual environment named as semeval2026t2 with `python==3.12.2` and install the required packages specified in `requirements.txt`

The script need to have access to the internet and download huggingface models from `cardiffnlp/twitter-roberta-base-sentiment-latest`



### Runnable
Simply run `bash.sh` to reproduce the results from end-to-end. The script will automatically set up the environment, install required packages, download pre-trained RoBERTa, finetune and make prediction on all three subtasks. The output prediction will be stored in `results/`. The finetuned model for each subtask will be stored in `model/`

The entire time for re-producing the experiment is less than 1-hour, tested on our local server powered by NVIDIA RTX 3500 Ada Generation 12GB memo.


### Parameters
The epoch, learning rate, batch, context length and other parameter settings are all provided in `run.sh` for each task.


### System description
A simplified system description is specified in `description.pdf`