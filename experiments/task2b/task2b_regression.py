# -*- coding: utf-8 -*-
"""
Gemma3 Dispositional Change (Δ Valence & Arousal) Forecasting
Phase-level prediction using previous phase(s) context.
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM
from datasets import Dataset

# -----------------------------
# Model Definition
# -----------------------------
class Gemma3ForRegression(nn.Module):
    def __init__(self, model_name, freeze_gemma3=True):
        super().__init__()
        self.gemma3 = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        hidden_size = self.gemma3.config.hidden_size
        self.regression_head = nn.Linear(hidden_size, 2)  # ΔValence & ΔArousal

        if freeze_gemma3:
            for param in self.gemma3.parameters():
                param.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.gemma3.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1)  # mean pooling

        preds = self.regression_head(pooled)
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(preds, labels)
        return {"loss": loss, "logits": preds}

# -----------------------------
# Dispositional Change Pairs
# -----------------------------
def build_dispositional_change_pairs(df, context="both", split="train"):
    pairs = []
    for user_id, user_df in df.groupby("user_id"):
        user_df = user_df.sort_values("group")
        phase_means = (
            user_df.groupby("group")[["valence", "arousal"]]
            .mean()
            .sort_index()
            .reset_index()
        )

        if split == "train":
            for i in range(len(phase_means) - 1):
                mean_1 = phase_means.loc[i]
                mean_2 = phase_means.loc[i + 1]
                group_1_df = user_df[user_df["group"] == mean_1["group"]]

                if context == "text":
                    input_text = " ".join(group_1_df["context"].dropna().astype(str).str.strip())
                elif context == "feelings":
                    input_text = " ".join(group_1_df["feelings"].dropna().astype(str).str.strip())
                elif context == "both":
                    input_text = (
                        " ".join(group_1_df["context"].dropna().astype(str).str.strip()) + " | " +
                        " ".join(group_1_df["feelings"].dropna().astype(str).str.strip())
                    )
                else:
                    raise ValueError("context must be one of ['text', 'feelings', 'both']")

                delta_valence = mean_2["valence"] - mean_1["valence"]
                delta_arousal = mean_2["arousal"] - mean_1["arousal"]

                pairs.append({
                    "user_id": user_id,
                    "group_from": mean_1["group"],
                    "group_to": mean_2["group"],
                    "prev_mean_valence": mean_1["valence"],
                    "prev_mean_arousal": mean_1["arousal"],
                    "curr_mean_valence": mean_2["valence"],
                    "curr_mean_arousal": mean_2["arousal"],
                    "delta_valence": delta_valence,
                    "delta_arousal": delta_arousal,
                    "input_text": input_text
                })
        else:  # test
            if len(phase_means) < 2:
                continue
            group_1_df = user_df
            mean_1_valence = group_1_df["valence"].mean()
            mean_1_arousal = group_1_df["arousal"].mean()
            mean_2 = phase_means.iloc[1]

            if context == "text":
                input_text = " ".join(group_1_df["context"].dropna().astype(str).str.strip())
            elif context == "feelings":
                input_text = " ".join(group_1_df["feelings"].dropna().astype(str).str.strip())
            elif context == "both":
                input_text = (
                    " ".join(group_1_df["context"].dropna().astype(str).str.strip()) + " | " +
                    " ".join(group_1_df["feelings"].dropna().astype(str).str.strip())
                )
            else:
                raise ValueError("context must be one of ['text', 'feelings', 'both']")

            delta_valence = mean_2["valence"] - mean_1_valence
            delta_arousal = mean_2["arousal"] - mean_1_arousal

            pairs.append({
                "user_id": user_id,
                "group_from": "all_previous",
                "group_to": mean_2["group"],
                "prev_mean_valence": mean_1_valence,
                "prev_mean_arousal": mean_1_arousal,
                "curr_mean_valence": mean_2["valence"],
                "curr_mean_arousal": mean_2["arousal"],
                "delta_valence": delta_valence,
                "delta_arousal": delta_arousal,
                "input_text": input_text
            })
    return pd.DataFrame(pairs)

# -----------------------------
# Dataset Preprocessing
# -----------------------------
def preprocess_dataset_for_regression_task2b(df, tokenizer, max_length=1024):
    task_instruction = (
        "You are an emotion analysis assistant. "
        "Given the observed emotional state (valence/arousal) and context, "
        "predict the average change in the next context."
    )
    combined_texts = []
    for _, row in df.iterrows():
        observed_valence_avg = row.get("prev_mean_valence", 0.0)
        observed_arousal_avg = row.get("prev_mean_arousal", 0.0)
        observed_texts = str(row.get("input_text", ""))
        prompt = (
            f"{task_instruction}\n"
            f"Observed Valence={observed_valence_avg:.3f}, Observed Arousal={observed_arousal_avg:.3f}\n"
            f"Observed Context: {observed_texts.strip()}"
        )
        combined_texts.append(prompt)

    labels = df[['delta_valence', 'delta_arousal']].values.astype(float)
    encodings = tokenizer(
        combined_texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    encodings['labels'] = torch.tensor(labels, dtype=torch.float)
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
# Prediction
# -----------------------------
def predict_dispositional_changes(model, tokenizer, df_pairs, max_length=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    task_instruction = (
        "You are an emotion analysis assistant. "
        "Given the observed emotional state (valence/arousal) and context, "
        "predict the average change in the next context."
    )

    results = []
    for _, row in tqdm(df_pairs.iterrows(), total=len(df_pairs), desc="Predicting Δavg changes"):
        observed_valence_avg = row.get("prev_mean_valence", 0.0)
        observed_arousal_avg = row.get("prev_mean_arousal", 0.0)
        observed_texts = str(row.get("input_text", ""))

        prompt = (
            f"{task_instruction}\n"
            f"Observed Valence={observed_valence_avg:.3f}, Observed Arousal={observed_arousal_avg:.3f}\n"
            f"Observed Context: {observed_texts.strip()}"
        )

        inputs = tokenizer(prompt, truncation=True, padding=True,
                           max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        delta_preds = outputs["logits"].cpu().numpy()[0]

        results.append({
            "pred_delta_valence": float(delta_preds[0]),
            "pred_delta_arousal": float(delta_preds[1])
        })
    return results

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Gemma3 Dispositional Change Δ Valence/Arousal")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-3-270m-it")
    parser.add_argument("--train_data_file", type=str, required=True)
    parser.add_argument("--test_data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./gemma3_state_change")
    parser.add_argument("--context", type=str, default="both", choices=["text", "feelings", "both"])
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)       # ✅ longer training
    parser.add_argument("--lr", type=float, default=1e-3)       # ✅ higher LR for head
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_df = pd.read_csv(args.train_data_file)
    test_df = pd.read_csv(args.test_data_file)

    train_pairs = build_dispositional_change_pairs(train_df, args.context, split="train")
    test_pairs = build_dispositional_change_pairs(test_df, args.context, split="test")

    train_split = int(len(train_pairs) * 0.8)
    train_dataset = preprocess_dataset_for_regression_task2b(train_pairs.iloc[:train_split], tokenizer, args.max_length)
    eval_dataset = preprocess_dataset_for_regression_task2b(train_pairs.iloc[train_split:], tokenizer, args.max_length)

    model = Gemma3ForRegression(args.model_name, freeze_gemma3=True)

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/task2b_{args.context}_model",
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
        fp16=torch.cuda.is_available(),
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
    # best_model_path = f"{args.output_dir}/task2b_{args.context}_best_model"
    # trainer.save_model(best_model_path)
    # tokenizer.save_pretrained(best_model_path)
    # print(f"Best model saved to {best_model_path}")

    preds = predict_dispositional_changes(trainer.model, tokenizer, test_pairs, args.max_length)
    results_df = pd.DataFrame({
        "user_id": test_pairs["user_id"],
        "pred_dispo_change_valence": [p["pred_delta_valence"] for p in preds],
        "pred_dispo_change_arousal": [p["pred_delta_arousal"] for p in preds]
    })
    results_df.to_csv(f"{args.output_dir}/task2b_{args.context}_test_predictions.csv", index=False)
    print(f"Predictions saved to {args.output_dir}/task2b_{args.context}_test_predictions.csv")

if __name__ == "__main__":
    main()
