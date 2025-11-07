# SemEval - 2026 Task 2: Predicting Variation in Emotional Valence and Arousal over Time from Ecological Essays

## Dataset

The split dataset and their feeling included version is in [teams channel](https://csiroau.sharepoint.com/:f:/r/sites/SemEval2026-Task2/Shared%20Documents/SemEval2026-Task2-Channel/feeling_dataset?csf=1&web=1&e=PsHcuk)
## TAKS1 

### Zero & Few shot
#### Ollama Inference Job

#### Overview
Run Ollama LLM inference on a GPU node using an Apptainer (`.sif`) container.

#### Build singularity
```
singularity build ollama_latest.sif docker://ollama/ollama:latest
```

#### Files 
- `ollama_latest.sif` – Apptainer container with Ollama  
- `task1_inference.py` – Python inference script  
- `config.json` – Configuration for Python script  
- `run.sh` - Job script
  
#### Run
```bash
source run.sh
```

#### Evaluation
```bash
python evaluation.py \
    --base_dir "prediction/gpt-oss:120b/zero_shot" \
    --file_template "test_cv{n}_low_feelings_general_pred.csv"
```


