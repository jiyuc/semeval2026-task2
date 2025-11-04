# Ollama Inference Job

## Overview
Run Ollama LLM inference on a GPU node using an Apptainer (`.sif`) container.

## Files
- `ollama_job.sbatch` – SLURM job script  
- `ollama_latest.sif` – Apptainer container with Ollama  
- `task1_inference.py` – Python inference script  
- `config.json` – Configuration for Python script  

## Usage
1. Ensure the `.sif` file is available.  
2. Set the model name in `ollama_job.sbatch` (`MODEL_NAME`).  
3. Submit the job:
```bash
sbatch ollama_job.sbatch
