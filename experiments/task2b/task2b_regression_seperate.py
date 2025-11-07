# -*- coding: utf-8 -*-
"""
Functional Gemma3 State Change (Δ Valence & Arousal) Forecasting
Separate models for ΔValence and ΔArousal
Using dispositional change where t = half the number of entries per user.
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
# Model Definitions
# -----------------------------
class Gemma3ForSingleRegression(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.gemma3 = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        hidden_size = self.gemma3.config.hidden_size
        self.regression_head = nn.Linear(hidden_size, 1)  # Only ΔValence

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.gemma3.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.last_hidden_state[:, -1, :]
        preds = self.regression_head(pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(preds.squeeze(-1), labels.squeeze(-1))

        return {"loss": loss, "logits": preds.squeeze(-1)}
        return {"loss": loss, "logits": preds}



# -----------------------------
# Dispositional Change Pairs
# -----------------------------
def build_state_change_pairs(df, context="both"):
    df = df.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)
    pairs = []

    for uid, user_df in df.groupby("user_id"):
        user_df = user_df.sort_values("timestamp").reset_index(drop=True)
        N = len(user_df)
        if N < 2:
            continue

        t = N // 2
        if t == 0:
            continue

        past_segment = user_df.iloc[:t]
        future_segment = user_df.iloc[t: 2*t]

        if len(future_segment) == 0:
            continue

        past_val = past_segment["valence"].mean()
        past_aro = past_segment["arousal"].mean()
        future_val = future_segment["valence"].mean()
        future_aro = future_segment["arousal"].mean()

        delta_val = future_val - past_val
        delta_aro = future_aro - past_aro

        pair = {
            "user_id": uid,
            "disposition_change_valence": delta_val,
            "disposition_change_arousal": delta_aro,
        }

        if context in ["text", "both"]:
            pair["prev_text"] = " ".join(list(past_segment["text"].astype(str)))
            pair["curr_text"] = " ".join(list(future_segment["text"].astype(str)))

        if context in ["feelings", "both"]:
            pair["prev_feelings"] = " | ".join(list(past_segment["feelings"].astype(str)))
            pair["curr_feelings"] = " | ".join(list(future_segment["feelings"].astype(str)))

        pairs.append(pair)

    return pd.DataFrame(pairs)


# -----------------------------
# Dataset Preprocessing
# -----------------------------
def preprocess_dataset(df, tokenizer, target, context="both", max_length=512):
    combined_texts = []

    for _, row in df.iterrows():
        prev_text = str(row.get("prev_text", ""))
        curr_text = str(row.get("curr_text", ""))
        prev_feelings = str(row.get("prev_feelings", "")) or prev_text
        curr_feelings = str(row.get("curr_feelings", "")) or curr_text

        if context == "text":
            combined_texts.append(f"Previous: {prev_text} || Current: {curr_text}")
        elif context == "feelings":
            combined_texts.append(f"Previous: {prev_feelings} || Current: {curr_feelings}")
        else:
            combined_texts.append(
                f"Previous: {prev_text} | Feelings: {prev_feelings} || "
                f"Current: {curr_text} | Feelings: {curr_feelings}"
            )

    labels = df[[target]].values.astype(float)
    enc = tokenizer(
        combined_texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    enc["labels"] = torch.tensor(labels, dtype=torch.float)
    return Dataset.from_dict(enc)


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
def predict_single_target(model, tokenizer, df_pairs, target, context="both", max_length=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    results = []

    for _, row in tqdm(df_pairs.iterrows(), total=len(df_pairs), desc=f"Predicting {target}"):

        if context == "text":
            text = f"Previous: {row.prev_text} || Current: {row.curr_text}"
        elif context == "feelings":
            text = f"Previous: {row.prev_feelings} || Current: {row.curr_feelings}"
        else:
            text = (
                f"Previous: {row.prev_text} | Feelings: {row.prev_feelings} "
                f"|| Current: {row.curr_text} | Feelings: {row.curr_feelings}"
            )

        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           padding=True, max_length=max_length).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        
        results.append({"logits": float(outputs["logits"].cpu().numpy()[0])})

    return results


# -----------------------------
# Evaluation (Single Target)
# -----------------------------
def evaluate_predictions(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    pearson, _ = pearsonr(true, pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "Pearson": pearson}


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-3-270m-it")
    parser.add_argument("--train_data_file", type=str, required=True)
    parser.add_argument("--test_data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./gemma3_state_change")
    parser.add_argument("--context", type=str, default="both")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load data
    train_df = pd.read_csv(args.train_data_file)
    test_df = pd.read_csv(args.test_data_file)

    # Build dispositional pairs
    train_pairs = build_state_change_pairs(train_df, args.context)
    test_pairs = build_state_change_pairs(test_df, args.context)

    # Split
    split = int(len(train_pairs) * 0.8)
    train_part = train_pairs.iloc[:split]
    eval_part = train_pairs.iloc[split:]
    predictions_dict = {}
    
    for target in ["disposition_change_valence", "disposition_change_arousal"]:
        print(f"\n--- Training model for {target} ---")
        train_dataset = preprocess_dataset(train_part, tokenizer, 
                                       target,
                                       args.context, args.max_length)
        eval_dataset = preprocess_dataset(eval_part, tokenizer, 
        target, args.context, args.max_length)

        model = Gemma3ForSingleRegression(args.model_name)

        training_args = TrainingArguments(
            output_dir=f"{args.output_dir}/task2b_{target}_{args.context}_seperate_model",
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
        best_model_path = f"{args.output_dir}/task2b_{target}_{args.context}_seperate_best_model"
        trainer.save_model(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        print(f"Best model for {target} saved to {best_model_path}")
        
        preds = predict_single_target(trainer.model, tokenizer, test_pairs, target, args.context, args.max_length)
        
            
            
        predictions_dict[target] = [p["logits"] for p in preds]


    # Build final results
    results_data = {
        "user_id": test_pairs["user_id"],
        "pred_disposition_change_valence": predictions_dict["disposition_change_valence"],
        "pred_disposition_change_arousal": predictions_dict["disposition_change_arousal"],
        "true_disposition_change_valence": test_pairs["disposition_change_valence"].tolist(),
        "true_disposition_change_arousal": test_pairs["disposition_change_arousal"].tolist()
    }

    results_df = pd.DataFrame(results_data)
    for var in ["disposition_change_valence", "disposition_change_arousal"]:
        true = results_df[f"true_{var}"]
        pred = results_df[f"pred_{var}"]
        metrics = evaluate_predictions(true, pred)
        print(f"\n[{var.upper()} Evaluation]")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    results_df.to_csv(f"{args.output_dir}/task2b_{args.context}_test_predictions_seperate.csv", index=False)
    print(f"Predictions saved to {args.output_dir}/task2b_{args.context}_test_predictions_seperate.csv")



if __name__ == "__main__":
    main()

