### Prerequisite and Config

Ensure the conda path is specified to `CONDA` in `run.sh`
Ensure the LLM model (Gemm3-270M) is stored in the same directory specified to `LLM_DIR` in the script `run.sh`. The augmnted files are also needs to be stored in `DATA_DIR` folder.


The script will automatically initialize a conda virtual environment named as semeval2026t2 and install the required packages specified in `requirements.txt`

The script need to have access to the internet and download huggingface model from `Google/Gemma-3-270M`



### Runnable
Simply run `run.sh` to reproduce the results from end-to-end. The script will automatically set up the environment, install required packages, download LLM, finetune and make prediction on all three subtasks. The finetuned models and the output predictions will be stored in `MODELS_DIR`.

The entire time for re-producing the experiment is less than 1-hour, tested on our local server powered by NVIDIA H100.


### Parameters
The epoch, learning rate, batch, context length and other parameter settings are all provided in training code for each task.


### System description
A simplified system description is specified in `description.pdf`