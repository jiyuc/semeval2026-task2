# -*- coding: utf-8 -*-
"""
Gemma3 Δ Valence & Arousal Forecasting with Learnable Time Embeddings
Uses previous + current text + time_delta bucket embedding
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
from tqdm import tqdm
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM
from datasets import Dataset

# -----------------------------
# Model Definition
# -----------------------------
class Gemma3ForRegression(nn.Module):
    def __init__(self, model_name, num_time_buckets=7, time_embedding_dim=8):
        super().__init__()
        self.gemma3 = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        hidden_size = self.gemma3.config.hidden_size
        self.time_embedding = nn.Embedding(num_time_buckets, time_embedding_dim)
        self.regression_head = nn.Linear(hidden_size + time_embedding_dim, 2)  # ΔValence & ΔArousal

    def forward(self, input_ids=None, attention_mask=None, labels=None, time_bucket=None):
        outputs = self.gemma3.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, -1, :]
        if time_bucket is not None:
            time_emb = self.time_embedding(time_bucket)
            pooled = torch.cat([pooled, time_emb], dim=1)
        preds = self.regression_head(pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(preds, labels)

        return {"loss": loss, "logits": preds}

# -----------------------------
# Dataset Preprocessing
# -----------------------------
def build_state_change_pairs(df, context="both", bins=None):
    """
    Build pairs of (previous, current) entries per user_id.
    Computes Δ valence/arousal and bucketed Δ time from timestamps.
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)
    pairs = []

    for uid, user_df in df.groupby("user_id"):
        user_df = user_df.sort_values("timestamp")
        for i in range(1, len(user_df)):
            prev_row = user_df.iloc[i - 1]
            curr_row = user_df.iloc[i]

            # Δ valence/arousal
            state_change_valence = curr_row["valence"] - prev_row["valence"]
            state_change_arousal = curr_row["arousal"] - prev_row["arousal"]

            # Δ time in seconds
            time_delta = (curr_row['timestamp'] - prev_row['timestamp']).total_seconds()

            # Bucket time
            time_bucket = np.digitize(time_delta, bins) - 1  # digitize starts at 1
            time_bucket = max(0, min(time_bucket, len(bins)-2))  # ensure valid bucket

            pair = {
                "user_id": uid,
                "state_change_valence": state_change_valence,
                "state_change_arousal": state_change_arousal,
                "time_bucket": time_bucket
            }

            # Context fields
            if context == "text":
                pair["prev_text"] = str(prev_row.get("text", ""))
                pair["curr_text"] = str(curr_row.get("text", ""))
            elif context == "feelings":
                pair["prev_feelings"] = str(prev_row.get("feelings", ""))
                pair["curr_feelings"] = str(curr_row.get("feelings", ""))
            elif context == "both":
                pair["prev_text"] = str(prev_row.get("text", ""))
                pair["curr_text"] = str(curr_row.get("text", ""))
                pair["prev_feelings"] = str(prev_row.get("feelings", ""))
                pair["curr_feelings"] = str(curr_row.get("feelings", ""))
            else:
                raise ValueError("context must be one of ['text', 'feelings', 'both']")

            pairs.append(pair)

    return pd.DataFrame(pairs)

def preprocess_dataset_for_regression(df, tokenizer, context, max_length=512):
    """
    Tokenize previous + current entries for Δ regression.
    Supports context = 'text', 'feelings', or 'both'.
    Include prepare time bucket tensor.
    """
    combined_texts = []

    for _, row in df.iterrows():
        prev_text = str(row.get("prev_text", ""))
        curr_text = str(row.get("curr_text", ""))
        prev_feelings = str(row.get("prev_feelings", "")) or prev_text
        curr_feelings = str(row.get("curr_feelings", "")) or curr_text

        if context == "text":
            combined_texts.append(f"Previous: {prev_text.strip()} || Current: {curr_text.strip()}")
        elif context == "feelings":
            combined_texts.append(f"Previous: {prev_feelings.strip()} || Current: {curr_feelings.strip()}")
        elif context == "both":
            combined_texts.append(
                f"Previous: {prev_text.strip()} | Feelings: {prev_feelings.strip()} || "
                f"Current: {curr_text.strip()} | Feelings: {curr_feelings.strip()}"
            )
        else:
            raise ValueError("context must be one of ['text', 'feelings', 'both']")

    labels = df[['state_change_valence', 'state_change_arousal']].values.astype(float)
    time_bucket = df['time_bucket'].values.astype(int)

    encodings = tokenizer(
        combined_texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    encodings['labels'] = torch.tensor(labels, dtype=torch.float)
    encodings['time_bucket'] = torch.tensor(time_bucket, dtype=torch.long)
    return Dataset.from_dict(encodings)

# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(pred):
    preds = pred.predictions
    labels = pred.label_ids
    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)
    return {"mse": mse, "mae": mae, "r2": r2}

# -----------------------------
# Prediction Helper
# -----------------------------
def predict_state_changes(model, tokenizer, df_pairs, context="both", max_length=512, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    results = []
    for _, row in tqdm(df_pairs.iterrows(), desc="Predicting Δ state changes", total=len(df_pairs)):
        # Build input according to context
        if context == "text":
            prev = str(row.get("prev_text", ""))
            curr = str(row.get("curr_text", ""))
            text_input = f"Previous: {prev.strip()} || Current: {curr.strip()}"

        elif context == "feelings":
            prev = str(row.get("prev_feelings", "")) or str(row.get("prev_text", ""))
            curr = str(row.get("curr_feelings", "")) or str(row.get("curr_text", ""))
            text_input = f"Previous: {prev.strip()} || Current: {curr.strip()}"

        elif context == "both":
            prev_text = str(row.get("prev_text", ""))
            curr_text = str(row.get("curr_text", ""))
            prev_feelings = str(row.get("prev_feelings", "")) or prev_text
            curr_feelings = str(row.get("curr_feelings", "")) or curr_text
            text_input = f"Previous: {prev_text.strip()} | Feelings: {prev_feelings.strip()} || Current: {curr_text.strip()} | Feelings: {curr_feelings.strip()}"

        else:
            raise ValueError("context must be one of ['text', 'feelings', 'both']")

        inputs = tokenizer(text_input, max_length=max_length, return_tensors="pt", truncation=True, padding=True).to(device)
        time_bucket_tensor = torch.tensor([row['time_bucket']], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(**inputs, time_bucket=time_bucket_tensor)
        preds = outputs["logits"].cpu().numpy()[0]
        results.append({
            "pred_state_change_valence": float(preds[0]),
            "pred_state_change_arousal": float(preds[1])
        })
    return results

# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_predictions(df):
    results = {}
    for var in ["state_change_valence", "state_change_arousal"]:
        true = df[f"true_{var}"].dropna().astype(float)
        pred = df.loc[true.index, f"pred_{var}"].astype(float)

        mse = mean_squared_error(true, pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred)
        pearson, _ = pearsonr(true, pred)

        bins = 5
        true_bins = pd.cut(true, bins=bins, labels=False)
        pred_bins = pd.cut(pred, bins=bins, labels=False)
        f1 = f1_score(true_bins, pred_bins, average="weighted")

        results[var] = {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Pearson": pearson,
            "Weighted F1": f1
        }
    return results

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Gemma3 Δ Valence/Arousal Forecasting with Time Embeddings")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-3-270m-it")
    parser.add_argument("--train_data_file", type=str, required=True)
    parser.add_argument("--test_data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./gemma3_state_change")
    parser.add_argument("--context", type=str, default="both", choices=["text", "feelings", "both"])
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--time_embedding_size", type=int, default=32)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_df = pd.read_csv(args.train_data_file)
    test_df = pd.read_csv(args.test_data_file)

    # Define time buckets (seconds)
    time_bins = [0, 60, 300, 900, 3600, 86400, 604800]  # 1min,5min,15min,1h,1d,1 week

    # Build pairs and compute Δ
    train_pairs = build_state_change_pairs(train_df, args.context, bins=time_bins)
    test_pairs = build_state_change_pairs(test_df, args.context, bins=time_bins)

    # Split train/val
    train_split = int(len(train_pairs) * 0.8)
    train_dataset = preprocess_dataset_for_regression(train_pairs.iloc[:train_split], tokenizer, args.context, args.max_length)
    eval_dataset = preprocess_dataset_for_regression(train_pairs.iloc[train_split:], tokenizer, args.context, args.max_length)

    model = Gemma3ForRegression(args.model_name, num_time_buckets=len(time_bins)-1, time_embedding_dim=args.time_embedding_size)

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/task2a_{args.context}_time_model",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="r2",
        greater_is_better=True,
        fp16=True,
        report_to="none",
        save_safetensors=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    best_model_path = f"{args.output_dir}/task2a_{args.context}_time_best_model"
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    print(f"Best model saved to {best_model_path}")

    # Predictions
    preds = predict_state_changes(model, tokenizer, test_pairs, args.context, args.max_length)

    # Build results dataframe
    results_data = {
        "user_id": test_pairs["user_id"],
        "pred_state_change_valence": [p["pred_state_change_valence"] for p in preds],
        "pred_state_change_arousal": [p["pred_state_change_arousal"] for p in preds],
        "true_state_change_valence": test_pairs["state_change_valence"].tolist(),
        "true_state_change_arousal": test_pairs["state_change_arousal"].tolist()
    }
    if args.context in ["text", "both"]:
        results_data["prev_text"] = test_pairs["prev_text"]
        results_data["curr_text"] = test_pairs["curr_text"]

    results_df = pd.DataFrame(results_data)

    metrics = evaluate_predictions(results_df)
    print("\n===== Δ State Change Evaluation Metrics =====")
    for target, values in metrics.items():
        print(f"\n[{target.upper()}]")
        for k, v in values.items():
            print(f"{k}: {v:.4f}")

    results_df.to_csv(f"{args.output_dir}/task2a_{args.context}_time_test_predictions.csv", index=False)
    print(f"Predictions saved to {args.output_dir}/task2a_{args.context}_time_test_predictions.csv")


if __name__ == "__main__":
    main()

