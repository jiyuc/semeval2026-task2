import argparse
import json
import os
import time
import re
import random
import pandas as pd
import requests
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# -----------------------------
# Helper Functions
# -----------------------------

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return json.load(f)

def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df

def simple_tokenize(text):
    return re.findall(r'\b\w+\b', str(text).lower())

def format_examples(examples):
    formatted = []
    for _, row in examples.iterrows():
        text_escaped = row["text"].replace('"', '\\"')
        formatted.append(
            f'Input: "{text_escaped}"\nOutput: {{"{{valence}}": {row["valence"]}, "{{arousal}}": {row["arousal"]}}}'
        )
    return "\n\n".join(formatted)

def get_bm25_examples(train_df, query_text, k=3):
    corpus = [simple_tokenize(t) for t in train_df['text'].tolist()]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(simple_tokenize(query_text))
    top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    examples = train_df.iloc[top_k]
    return format_examples(examples)

def call_model_api(url, model, prompt, think, num_ctx, retries, wait_time):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": False,
        "options": {"num_ctx": num_ctx, "think": think}
    }

    for attempt in range(retries):
        try:
            response = requests.post(url, data=json.dumps(payload))
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(wait_time)
    return ""

def parse_prediction(content):
    val = re.search(r'"valence":\s*([-+]?\d+\.?\d*)', content)
    aro = re.search(r'"arousal":\s*([-+]?\d+\.?\d*)', content)
    return {
        "valence": float(val.group(1)) if val else 0.0,
        "arousal": float(aro.group(1)) if aro else 1.0
    }


# -----------------------------
# Prediction Logic
# -----------------------------
def predict(text, mode, config, model, think, train_df=None):
    num_ctx = config.get("num_ctx", 500)
    few_k = config["few_shot"]["k"]
    seed = config["few_shot"]["seed"]

    base_prompt = config["prompt"]["base"]  # <- Base prompt reused

    if mode == "zero_shot":
        template = config["prompt"]["zero_shot"]
        prompt = f"{base_prompt}\n\n" + template.format(text=text)

    elif mode == "few_shot_random":
        few = train_df.sample(n=few_k, random_state=seed)
        examples = format_examples(few)
        template = config["prompt"]["few_shot_random"]
        prompt = f"{base_prompt}\n\n" + template.format(text=text, examples=examples)

    elif mode == "few_shot_bm25":
        examples = get_bm25_examples(train_df, text, k=few_k)
        template = config["prompt"]["few_shot_bm25"]
        prompt = f"{base_prompt}\n\n" + template.format(text=text, examples=examples)

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    content = call_model_api(config["url"], model, prompt, think, num_ctx, config["max_retries"], config["wait_time"])
    return parse_prediction(content)

def run_predictions(df_test, df_train, mode, config, model, think):
    results = []
    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc=f"{mode} | {model} | {think}"):
        pred = predict(row["text"], mode, config, model, think, train_df=df_train)
        results.append({
            "text": row["text"],
            "pred_valence": pred["valence"],
            "pred_arousal": pred["arousal"],
            "true_valence": row.get("valence"),
            "true_arousal": row.get("arousal")
        })
    return pd.DataFrame(results)


# -----------------------------
# Main Runner
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Emotion prediction with config-based few-shot/zero-shot settings")
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    args = parser.parse_args()

    config = load_config(args.config)
    random.seed(config["few_shot"]["seed"])

    for mode in config["modes"]:
        for model in config["models"]:
            for think in config["thinking"]:
                for fold in config["data"]["folds"]:
                    print(f"\nRunning {mode} | Model={model} | Think={think} | Fold={fold}")

                    test_path = config["data"]["test_template"].format(fold)
                    train_path = config["data"]["train_template"].format(fold)

                    df_test = load_data(test_path)
                    df_train = load_data(train_path) if mode != "zero_shot" else None

                    pred_df = run_predictions(df_test, df_train, mode, config, model, think)

                    # Determine few-shot type string
                    if mode == "zero_shot":
                        few_shot_type = "none"
                    elif mode == "few_shot_random":
                        few_shot_type = "random"
                    elif mode == "few_shot_bm25":
                        few_shot_type = "bm25"

                    # Build output directory and path
                    output_dir = f"{config['data']['output_dir']}/{model}/{mode}_{few_shot_type}_{think}/"
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = f"{output_dir}fold_{fold}_pred.csv"

                    pred_df.to_csv(output_path, index=False)
                    print(f"Saved predictions to: {output_path}")

if __name__ == "__main__":
    main()
