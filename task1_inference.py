# Python 3.12
# Inference script for valence/arousal regression using a fine-tuned (XLM-)RoBERTa-base model.
# - Loads a saved model directory (from task1 fine-tuning)
# - Prepares inputs exactly like training (single-text inputs)
# - Predicts and writes integer-rounded predictions back to the CSV under the requested target column.

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


def _ensure_cols(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")


def _default_out_path(input_csv: str, target: str) -> str:
    base, ext = os.path.splitext(input_csv)
    return f"{base}.{target}.predicted{ext}"


def _predict(
    texts: List[str],
    model_dir: str,
    batch_size: int = 32,
    max_length: int = 512,
    device: str | None = None,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=1)
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    model.to(device_t)

    preds_all: List[float] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device_t) for k, v in enc.items()}
            out = model(**enc)
            logits = out.logits.squeeze(-1).detach().cpu().numpy()
            if logits.ndim == 0:
                logits = np.array([float(logits)], dtype=np.float32)
            preds_all.extend(logits.tolist())

    # round to closest integer and cast to small int
    # preds_all = MinMaxScaler(feature_range=(-2.0, 2.0)).fit_transform(preds_all).flatten()
    return np.asarray([round(n, 0) for n in preds_all], dtype=np.int8)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a fine-tuned RoBERTa-base regression model and write predictions back to CSV."
    )
    parser.add_argument("--csv", required=True, help="Path to input CSV (same schema as training).")
    parser.add_argument("--model_dir", required=True, help="Directory containing the fine-tuned model/tokenizer.")
    parser.add_argument("--target",
                        choices=["valence", "arousal"],
                        required=True,
                        help="Which target to predict; determines destination column name to write.")
    parser.add_argument("--text_col", default="text",
                        help="Name of text column if different from 'text'.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_csv", default=None,
                        help="Optional output CSV path; if omitted, writes alongside the input with a suffix.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Ensure required columns exist to mirror training schema for task1
    _ensure_cols(df, [args.text_col])#, "user_id", "text_id", "timestamp"])

    texts = df[args.text_col].fillna("").astype(str).tolist()
    if len(texts) == 0:
        # still ensure column exists and save out
        if args.target not in df.columns:
            df[args.target] = np.nan
        out_path = args.output_csv or _default_out_path(args.csv, args.target)
        df.to_csv(out_path, index=False)
        print(f"No rows to predict. Wrote CSV to {out_path}")
        return

    preds = _predict(
        texts=texts,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Create/overwrite the destination column with predictions where missing; keep any existing non-NaNs
    if args.target not in df.columns:
        df[args.target] = np.nan

    mask = df[args.target].isna()
    df.loc[mask, args.target] = preds[mask.values]  # align by row order

    out_path = args.output_csv or _default_out_path(args.csv, args.target)
    df.round(2).to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    main()