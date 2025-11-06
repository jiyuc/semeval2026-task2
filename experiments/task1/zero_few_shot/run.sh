#!/bin/bash
#SBATCH --job-name=ollama_job
#SBATCH --output=ollama_%j.log
#SBATCH --error=ollama_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1   
#SBATCH --time=04:00:00   
#SBATCH --mem=32G          
#SBATCH -A OD-240335

# Load modules
module load apptainer
module load cuda       # if needed for GPU

# Set directories
SCRATCH_DIR=/scratch3/bol107/table_generation_eacl

export OLLAMA_HOST="http://127.0.0.1:11434"

# Ensure model directory exists
export OLLAMA_MODELS="$SCRATCH_DIR/ollama_models"
mkdir -p "$OLLAMA_MODELS"

echo "===================================="
echo " Starting Ollama server"
echo "===================================="
nohup apptainer exec --nv -B "$SCRATCH_DIR:/olla_bin" --env OLLAMA_MODELS="$SCRATCH_DIR/ollama_models" ollama_latest.sif ollama serve > "$SCRATCH_DIR/ollama_server.log" 2>&1 &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for Ollama server to initialize..."
sleep 25   # increase if model is large

echo "===================================="
echo " Checking if model exists"
echo "===================================="
MODELS=("gemma3:12b" "gemma3:27b")

for MODEL_NAME in "${MODELS[@]}"; do
    # Check if the model exists (by name) via running server
    apptainer exec --nv -B "$SCRATCH_DIR:/olla_bin" --env OLLAMA_MODELS="$SCRATCH_DIR/ollama_models" ollama_latest.sif ollama list | grep -q "$MODEL_NAME"

    if [ $? -ne 0 ]; then
        echo "Model $MODEL_NAME not found — pulling..."
        apptainer exec --nv -B "$SCRATCH_DIR:/olla_bin" --env OLLAMA_MODELS="$SCRATCH_DIR/ollama_models" ollama_latest.sif ollama pull "$MODEL_NAME"
    else
        echo "Model $MODEL_NAME already exists."
    fi
done

# Activate conda environment
conda activate necva_env

python task1_inference.py --config config.json
# ---------------- CLEANUP ----------------
echo "===================================="
echo " Stopping Ollama server..."
echo "===================================="
kill $SERVER_PID 2>/dev/null || pkill -f "ollama_latest.sif serve"

echo "===================================="
echo " All datasets processed successfully."
echo "===================================="
