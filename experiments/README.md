# Ollama Inference Job

## Overview
Run Ollama LLM inference on a GPU node using an Apptainer (`.sif`) container.

## Build singularity
```
singularity build ollama_latest.sif docker://ollama/ollama:latest
```

## TAKS1 

### Files 
- `ollama_latest.sif` – Apptainer container with Ollama  
- `task1_inference.py` – Python inference script  
- `config.json` – Configuration for Python script  
- `run.sh` - Job script
  
### Run
```bash
sbatch run.sh
```

### Evaluation
```bash
python your_script.py \
    --base_dir "prediction/gpt-oss:120b/zero_shot" \
    --file_template "test_cv{n}_low_feelings_general_pred.csv"
```
