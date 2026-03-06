# CSIRO-LT at SemEval-2026 Task 2: *In-the-Wild* Valence and Arousal Forecasting on Ecological Text Time Series

This repository contains the code and data for our submission to SemEval 2026 Task 2. The project focuses on valence and arousal regression across three subtasks:
1.  **Subtask 1**: Predicting the current valence and arousal of a user based on their social media post and context.
2.  **Subtask 2a**: Forecasting user state change (valence and arousal).
3.  **Subtask 2b**: Forecasting user disposition change (long-term valence and arousal trends).

## Repository Structure

The project is organized into several key directories. The README under each directory provides detailed information about the code and data within that directory.

-   `llm_capsule/`: (contributed by Necva) Contains the implementation using Large Language Models (specifically Google's Gemma-3-270M).
-   `roberta_capsule/`: (contributed by Jiyu) Contains the implementation using the RoBERTa model (based on `cardiffnlp/twitter-roberta-base-sentiment-latest`).
-   `TRAIN_RELEASE_3SEP2025/`: (provided by organizer) Contains the official training data, frequency analysis, and visualizations of the dataset.

## Sub-Modules

### 1. LLM Capsule (`llm_capsule/`)

This module uses Google's Gemma-3-270M for regression tasks. It involves fine-tuning the LLM with context-augmented data.

-   **Key Files**:
    -   `task1_regression.py`, `task2a_regression.py`, `task2b_regression.py`: Training scripts for each subtask.
    -   `run.sh`: End-to-end script to set up the environment, download the model, and run experiments.
    -   `requirements.txt` & `environment.yml`: Dependency lists.

### 2. RoBERTa Capsule (`roberta_capsule/`)

This module uses a sentiment-aware RoBERTa model for predicting valence and arousal.

-   **Key Files**:
    -   `task1_trainer.py`, `task2a_trainer.py`, `task2b_trainer.py`: Fine-tuning scripts.
    -   `task1_predictor.py`, `task2a_predictor.py`, `task2b_predictor.py`: Inference scripts.
    -   `run.sh`: End-to-end script for environment setup, data splitting, training, and prediction.
    -   `create_cv_split.py`: Script to generate cross-validation splits.

## Getting Started

### Prerequisites

-   Conda (Anaconda or Miniconda)
-   Python 3.12+
-   CUDA-enabled GPU (tested on NVIDIA H100 and RTX 3500 Ada)
-   Internet connection (to download pre-trained models from Hugging Face)

### Running the Experiments

Each capsule has its own `run.sh` script that automates the setup and execution:

#### For LLM-based approach:
```bash
cd llm_capsule
bash run.sh
```

#### For RoBERTa-based approach:
```bash
cd roberta_capsule
bash run.sh
```

*Note: Ensure you update the configuration paths (e.g., `CONDA` path) in the respective `run.sh` files if necessary.*

## Data

The official dataset provided in `TRAIN_RELEASE_3SEP2025/` includes:
-   `train_subtask1.csv`
-   `train_subtask2a.csv`
-   `train_subtask2b.csv`
-   Additional detailed files for user disposition and state changes.

Visualizations of the data can be found in `TRAIN_RELEASE_3SEP2025/dist_valence_arousal_by_top25_feeling_words.pdf`.

## License

This project is licensed under the terms of the LICENSE file included in the root directory.
