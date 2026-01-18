# -*- coding: utf-8 -*-
"""
Functional Gemma3 State Change (Δ Valence & Arousal) Forecasting
Using both previous and current texts as context
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM
)

from datasets import Dataset


# -----------------------------
# Model Definition
# -----------------------------
class Gemma3ForRegression(nn.Module):
    def __init__(self, model_name, freeze_gemma3=True):
        super().__init__()
        self.gemma3 = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto"
        )
        hidden_size = self.gemma3.config.hidden_size

        self.valence_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Tanh()       # [-1, 1]
        )

        self.arousal_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()    # [0, 1]
        )

        if freeze_gemma3:
            for param in self.gemma3.parameters():
                param.requires_grad = False


    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.gemma3.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Better pooling than last token
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)

        pooled = (hidden * mask).sum(1) / mask.sum(1)

        valence = self.valence_head(pooled) * 2     # [-2, 2]
        arousal = self.arousal_head(pooled) * 2     # [0, 2]

        preds = torch.cat([valence, arousal], dim=1)

        loss = None
        if labels is not None:
            loss = nn.MSELoss()(preds, labels)

        return {"loss": loss, "logits": preds}


# -----------------------------
# Dataset Preprocessing
# -----------------------------
def build_state_pairs_with_history(df, context="both", split="train", history_len=5):
    """
    Adaptive history:
    - Use min(history_len, available_history)
    - Causal: train prev=t-2 target=t-1, test prev=t-1
    - History contains numeric deltas
    """
    df = df.sort_values(
        ["user_id", "collection_phase", "timestamp"]
    ).reset_index(drop=True)

    pairs = []

    for user_id, user_df in df.groupby("user_id"):
        phases = list(user_df.groupby("collection_phase"))

        # TEST → only last collection phase
        if split == "test":
            phases = [max(phases, key=lambda x: x[0])]

        for phase_id, phase_df in phases:
            phase_df["timestamp"] = pd.to_datetime(
                phase_df["timestamp"], dayfirst=True, errors="coerce"
            )
            phase_df = phase_df.sort_values("timestamp")

            if split in ["train", "validation"]:
                if len(phase_df) < 3:
                    continue
                history_pool = phase_df.iloc[:-2]
                prev_row = phase_df.iloc[-2]
                target_row = phase_df.iloc[-1]
            else:
                if len(phase_df) < 2:
                    continue
                history_pool = phase_df.iloc[:-1]
                prev_row = phase_df.iloc[-1]
                target_row = None

            # Adaptive history
            history_df = history_pool.iloc[-history_len:]

            history_va = []
            prev_val, prev_ar = None, None
            for _, row in history_df.iterrows():
                if prev_val is not None:
                    delta_v = row.valence - prev_val
                    delta_a = row.arousal - prev_ar
                    history_va.append(f"({row.valence:.2f}, {row.arousal:.2f}, Δv={delta_v:.2f}, Δa={delta_a:.2f})")
                else:
                    history_va.append(f"({row.valence:.2f}, {row.arousal:.2f}, Δv=0.00, Δa=0.00)")
                prev_val, prev_ar = row.valence, row.arousal

            item = {
                "user_id": user_id,
                "collection_phase": phase_id,
                "history_va": " → ".join(history_va),
                "prev_valence": prev_row.valence,
                "prev_arousal": prev_row.arousal,
            }

            if target_row is not None:
                item["valence"] = target_row.valence
                item["arousal"] = target_row.arousal

            if context in ["text", "both"]:
                item["text"] = str(prev_row.get("context", ""))
            if context in ["feelings", "both"]:
                item["feelings"] = str(prev_row.get("feelings", ""))

            pairs.append(item)

    return pd.DataFrame(pairs)

def preprocess_dataset(df, tokenizer, context="both", max_length=512):
    instruction = (
        "You are an emotion forecasting model.\n"
        "Given the emotional trajectory of a person over time, "
        "predict the NEXT valence and arousal."
    )

    prompts = []

    for _, row in df.iterrows():
        prompt = (
            f"{instruction}\n\n"
            f"Emotional History (Valence, Arousal, Δvalence, Δarousal):\n"
            f"{row.history_va}\n\n"
            f"Most Recent State:\n"
            f"Valence={row.prev_valence:.3f}, "
            f"Arousal={row.prev_arousal:.3f}\n"
        )

        if context == "text":
            prompt += f"Context Text: {row.text}"
        elif context == "feelings":
            prompt += f"Self-Reported Feelings: {row.feelings}"
        else:
            prompt += (
                f"Context Text: {row.text} | "
                f"Self-Reported Feelings: {row.feelings}"
            )

        prompts.append(prompt)

    labels = df[["valence", "arousal"]].values.astype("float32")

    encodings = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    encodings["labels"] = torch.tensor(labels)
    return Dataset.from_dict(encodings)

# =========================================================
# Metrics
# =========================================================
def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids

    return {
        "mse": mean_squared_error(labels, preds),
        "mae": mean_absolute_error(labels, preds),
        "r2": r2_score(labels, preds)
    }

# =========================================================
# Test Prediction (Δ only)
# =========================================================
def predict_deltas(model, tokenizer, df, context="both", max_length=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    instruction = (
        "You are an emotion forecasting model.\n"
        "Given the emotional trajectory of a person over time, "
        "predict the NEXT valence and arousal."
    )

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        prompt = (
            f"{instruction}\n\n"
            f"Emotional History (Valence, Arousal, Δvalence, Δarousal):\n"
            f"{row.history_va}\n\n"
            f"Most Recent State:\n"
            f"Valence={row.prev_valence:.3f}, "
            f"Arousal={row.prev_arousal:.3f}\n"
        )

        if context == "text":
            prompt += f"Context Text: {row.text}"
        elif context == "feelings":
            prompt += f"Self-Reported Feelings: {row.feelings}"
        else:
            prompt += (
                f"Context Text: {row.text} | "
                f"Self-Reported Feelings: {row.feelings}"
            )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(device)

        with torch.no_grad():
            preds = model(**inputs)["logits"][0].cpu().numpy()

        results.append({
            "pred_state_change_valence": preds[0] - row.prev_valence,
            "pred_state_change_arousal": preds[1] - row.prev_arousal
        })

    return pd.DataFrame(results)



# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="unsloth/gemma-3-270m-it")
    parser.add_argument("--train_data_file", required=True)
    parser.add_argument("--test_data_file", required=True)
    parser.add_argument("--output_dir", default="./gemma3_state_change")
    parser.add_argument("--context", choices=["text", "feelings", "both"], default="both")
    parser.add_argument("--history_len", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)       # ✅ longer training
    parser.add_argument("--lr", type=float, default=1e-3)       # ✅ higher LR for head
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_df = pd.read_csv(args.train_data_file)
    test_df = pd.read_csv(args.test_data_file)
    # test_df = test_df[test_df["is_forecasting_user"] == True]

    train_pairs = build_state_pairs_with_history(
        train_df, args.context, "train", args.history_len
    )
    test_pairs = build_state_pairs_with_history(
        test_df, args.context, "test", args.history_len
    )

    split = int(0.8 * len(train_pairs))
    train_ds = preprocess_dataset(
        train_pairs[:split], tokenizer, args.context, args.max_length
    )
    val_ds = preprocess_dataset(
        train_pairs[split:], tokenizer, args.context, args.max_length
    )

    model = Gemma3ForRegression(args.model_name)
    for param in model.gemma3.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/task2a_{args.context}_model",
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
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    best_model_path = f"{args.output_dir}/task2a_{args.context}_best_model" 
    trainer.save_model(best_model_path) 
    tokenizer.save_pretrained(best_model_path) 
    print(f"Best model saved to {best_model_path}")

    delta_preds = predict_deltas(
        trainer.model,
        tokenizer,
        test_pairs,
        args.context,
        args.max_length
    )

    results_df = pd.concat(
        [test_pairs[["user_id"]].reset_index(drop=True), delta_preds],
        axis=1
    )

    results_df.to_csv(f"{args.output_dir}/task2a_{args.context}_test_predictions.csv", index=False) 
    print(f"Predictions saved to {args.output_dir}/task2a_{args.context}_test_predictions.csv")

if __name__ == "__main__":
    main()
