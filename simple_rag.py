import pandas as pd
import evaluate
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch



# Load the training and test CSV files
train_data = pd.read_csv('data/split/subtask1_train_cv1.csv')
test_data = pd.read_csv('data/split/subtask1_test_cv1.csv')

# Concatenate text, valence, and arousal scores
train_data['text_with_scores'] = train_data['text'] + " | Valence: " + train_data[
    'valence'].astype(str) + " | Arousal: " + train_data['arousal'].astype(str)



# Setup local LLM
local_llm_path = "meta-llama/Llama-3.2-1B-Instruct"
# Load tokenizer from the local model path
tokenizer = AutoTokenizer.from_pretrained(local_llm_path)


bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    local_llm_path,
    config=bnb_config,  # Pass the BitsAndBytes config here
    device_map="cuda",
    # dtype=torch.bfloat16,  # Use FP16 precision for computation, can also use float32 or other types
)


Settings.llm = HuggingFaceLLM(
    # context_window=512,
    max_new_tokens=1,
    tokenizer_name=local_llm_path,
    # model_name=local_llm_path,
    device_map="cuda",
    model=model,
)

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5") # Use a small embedding model



# Create the index
documents = []
for _, row in train_data.iterrows():
    documents.append(
        Document(text=row['text_with_scores'], extra_info={'valence': row['valence'], 'arousal': row['arousal']}))

# Initialize the index
index = VectorStoreIndex.from_documents(documents)

# Store the index
index.storage_context.persist(persist_dir='train_index_storage')

# Querying and Evaluation
query_engine = index.as_query_engine(verbose=True)


def extract_score(text):
    try:
        return float(text)
    except Exception:
        return -10


predicted_valence = []
predicted_arousal = []

print("Starting querying...")
for _, row in test_data.iterrows():
    text = row['text']

    # Query Valence
    response_v = query_engine.query(f"Given the valence score in range -2 (negative) to 2 (positive) and the text: {text} The valence score of the text above is:")
    # Access retrieved documents from the response
    # # print(response_v.source_nodes)
    predicted_valence.append(extract_score(str(response_v)))

    # # Query Arousal
    response_a = query_engine.query(f"Given the arousal score in range 0 (low) to 2 (high) and the text: {text} The arousal score of the text above is:")
    predicted_arousal.append(extract_score(str(response_a)))

test_data['predicted_valence'] = predicted_valence
test_data['predicted_arousal'] = predicted_arousal

# Save results
test_data.to_csv('data/split/subtask1_test_cv1_with_predictions.csv', index=False)

# Evaluation metrics
mse_metric = evaluate.load("mse")
pearson_metric = evaluate.load("pearsonr")
r2_metric = evaluate.load("r_squared")
mae_metric = evaluate.load("mae")
rmse_metric = evaluate.load("mse")
f1_metric = evaluate.load("f1")


def evaluate_predictions(preds, refs, metric_name):
    mse = mse_metric.compute(predictions=preds, references=refs, squared=True)["mse"]
    pearson = pearson_metric.compute(predictions=preds, references=refs)["pearsonr"]
    r2 = r2_metric.compute(predictions=preds, references=refs)
    mae = mae_metric.compute(predictions=preds, references=refs)["mae"]
    rmse = rmse_metric.compute(predictions=preds, references=refs, squared=False)["mse"]
    f1 = \
    f1_metric.compute(predictions=[round(n) for n in preds], references=[round(n) for n in refs], average="weighted")[
        "f1"]

    print(f"Results for {metric_name}:")
    print(f"MSE: {mse}")
    print(f"Pearson: {pearson}")
    print(f"R2: {r2}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"F1: {f1}")


print("Evaluating Valence:")
evaluate_predictions(test_data['predicted_valence'], test_data['valence'], "Valence")

# print("Evaluating Arousal:")
evaluate_predictions(test_data['predicted_arousal'], test_data['arousal'], "Arousal")