# -*- coding: utf-8 -*-
"""
Functional Gemma3 Valence/Arousal Regression Fine-Tuning
Based on GEMMA3 classification setup, adapted for regression
"""

import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    AutoModelForCausalLM)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    f1_score
)
# -----------------------------
# Regression Model Wrapper
# -----------------------------
class Gemma3ForRegression(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # Load pretrained GEMMA3
        self.gemma3 = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        hidden_size = self.gemma3.config.hidden_size
        self.regression_head = nn.Linear(hidden_size, 2)  # valence & arousal

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.gemma3.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, hidden)
        pooled = last_hidden_state[:, -1, :]           # (batch, hidden)
        preds = self.regression_head(pooled)          # (batch, 2)

        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(preds, labels)

        return {"loss": loss, "logits": preds}


# -----------------------------
# Dataset Preprocessing
# -----------------------------
def preprocess_dataset_for_regression(df, tokenizer, context, max_length=512):
    if context == "text":
        texts = df['text'].fillna("").tolist()
    elif context == "feelings":
        # Use feelings if available; fallback to text
        texts = [
            f if isinstance(f, str) and f.strip() != "" else t
            for f, t in zip(df["feelings"], df["text"])
        ]

    elif context == "both":
        # Combine both (if available)
        texts = [
            f"Text: {t.strip()} | Feelings: {f.strip()}" if isinstance(f, str) and f.strip() != "" else t.strip()
            for f, t in zip(df["feelings"], df["text"])
        ]

    labels = df[['valence', 'arousal']].values.astype(float)
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    encodings['labels'] = torch.tensor(labels, dtype=torch.float)
    return Dataset.from_dict(encodings)


# -----------------------------
# Regression Metrics
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
def predict_valence_arousal(model, tokenizer, texts, max_length=512, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    results = []
    for text in tqdm(texts, desc="Predicting"):
        inputs = tokenizer(text, max_length=max_length, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        preds = outputs["logits"].cpu().numpy()[0]
        results.append({"valence": float(preds[0]), "arousal": float(preds[1])})
    return results

def evaluate_predictions(df):
    """
    Compute MSE, RMSE, MAE, R2, Pearson, and Weighted F1
    for valence and arousal predictions.
    """
    results = {}
    for var in ["valence", "arousal"]:
        true = df[f"true_{var}"].dropna().astype(float)
        pred = df.loc[true.index, f"pred_{var}"].astype(float)

        mse = mean_squared_error(true, pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred)
        pearson, _ = pearsonr(true, pred)

        # Weighted F1 (after discretizing into bins)
        bins = 5 if var == "valence" else 3  # adjustable granularity
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
# Main Function
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Gemma3 Valence/Arousal Regression Fine-Tuning")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-3-270m-it", help="Pretrained GEMMA3 model")
    parser.add_argument("--train_data_file", type=str, required=True, help="Path to training CSV dataset")
    parser.add_argument("--test_data_file", type=str, required=True, help="Path to test CSV dataset")
    parser.add_argument("--output_dir", type=str, default="./gemma3_regression", help="Output directory")
    parser.add_argument("--context", type=str, default="both", choices=["text", "feelings", "both"], help="Select which context to use for extraction: 'text', 'feelings', or 'both'.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load CSV datasets
    train_df = pd.read_csv(args.train_data_file)
    test_df = pd.read_csv(args.test_data_file)

    # Train/Validation Split
    train_split = int(len(train_df) * 0.8)
    train_dataset = preprocess_dataset_for_regression(train_df.iloc[:train_split], tokenizer, args.context, args.max_length)
    eval_dataset = preprocess_dataset_for_regression(train_df.iloc[train_split:], tokenizer, args.context, args.max_length)
    test_dataset = preprocess_dataset_for_regression(test_df, tokenizer, args.context, args.max_length)

    # Initialize model
    model = Gemma3ForRegression(args.model_name)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/{args.context}_model",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=50,
        eval_strategy="steps",     # evaluate every eval_steps
        eval_steps=50,             # must match save frequency ideally
        save_strategy="steps",     # save checkpoint every save_steps
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="r2",
        greater_is_better=True,
        fp16=True,
        report_to="none",
        save_safetensors=False
    )
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()
    best_model_path = f"{args.output_dir}/{args.context}_best_model"
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    print(f"Best model saved to {best_model_path}")

    # Predictions
    if args.context == "text":
        texts = test_df['text'].fillna("").tolist()
    elif args.context == "feelings":
        texts = [
        f if isinstance(f, str) and f.strip() != "" else t
        for f, t in zip(test_df["feelings"], test_df["text"])
       ] 
    elif args.context == "both":
        # Combine both (if available)
        texts = [
            f"Text: {t.strip()} | Feelings: {f.strip()}"
            if isinstance(f, str) and f.strip() != ""
            else t.strip()
            for f, t in zip(test_df["feelings"], test_df["text"])
        ]
    #texts = test_df['text'].tolist()
    preds = predict_valence_arousal(trainer.model, tokenizer, texts)
    #print(preds)
    results_df = pd.DataFrame({
        "text": texts,
        "pred_valence": [p["valence"] for p in preds],
        "pred_arousal": [p["arousal"] for p in preds],
        "true_valence": test_df["valence"].tolist(),
        "true_arousal": test_df["arousal"].tolist()
    })

    #results_df = results_df.dropna(subset=["pred_valence", "pred_arousal", "true_valence", "true_arousal"])
    metrics = evaluate_predictions(results_df)

    print("\n===== Evaluation Metrics =====")
    for target, values in metrics.items():
        print(f"\n[{target.upper()}]")
        for k, v in values.items():
            print(f"{k}: {v:.4f}")
            
    results_df.to_csv(f"{args.output_dir}/{args.context}_test_predictions.csv", index=False)
    print(f"Predictions saved to {args.output_dir}/{args.context}_test_predictions.csv")


if __name__ == "__main__":
    main()

