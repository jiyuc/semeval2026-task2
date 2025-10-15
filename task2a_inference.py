# Python 3.12
# Inference script for state-change regression using a fine-tuned RoBERTa-base model.
# - Loads the saved model dir (from task2a_trainer.py fine-tuning)
# - Reconstructs paired inputs the same way as training:
#     for each user_id: (text1, state1, text2, time_diff)
#   and concatenates into a single "text" field:
#     "{text1} {state1} {text2} {time_diff}"
# - Tokenizes, predicts, and writes predictions back to the original CSV under
#   either state_change_valence or state_change_arousal.

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


def _ensure_cols(df: pd.DataFrame, col_names: List[str]) -> None:
    missing = [c for c in col_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")


def _time_diff_seconds(ts1: str, ts2: str) -> float:
    # Expected format used in training: "%Y-%m-%d %H:%M:%S"
    # Let pandas parse to be robust.
    t1 = pd.to_datetime(ts1)
    t2 = pd.to_datetime(ts2)
    return float((t2 - t1).total_seconds())


def _build_pairs_for_user(group: pd.DataFrame,
                          feature_col: str) -> pd.DataFrame:
    # Sort by timestamp to match training
    group = group.sort_values(by=["timestamp"]).reset_index(drop=True)
    rows = []
    for i in range(1, len(group)):
        prev_row = group.iloc[i - 1]
        curr_row = group.iloc[i]
        rows.append({
            "user_id": prev_row["user_id"],
            "text1": str(prev_row["text"]) if pd.notna(prev_row["text"]) else "",
            "text2": str(curr_row["text"]) if pd.notna(curr_row["text"]) else "",
            "state1": prev_row[feature_col],
            "time_diff": _time_diff_seconds(prev_row["timestamp"], curr_row["timestamp"]),
            # Keep original text_id of the "current" row so we can merge predictions back
            "text_id": prev_row["text_id"],
        })
    return pd.DataFrame(rows)


def _prepare_inference_dataframe(df: pd.DataFrame, feature_col: str) -> pd.DataFrame:
    # Build consecutive pairs per user_id (same as training)
    parts = []
    for uid, group in df.groupby("user_id", sort=False):
        if len(group) < 2:
            continue
        parts.append(_build_pairs_for_user(group, feature_col))
    if not parts:
        return pd.DataFrame(columns=["user_id", "text_id", "text"])
    paired_df = pd.concat(parts, ignore_index=True)

    # Compose the single 'text' input as in training combine_text()
    paired_df["text"] = (
        paired_df["text1"].astype(str)
        + " "
        + paired_df["state1"].astype(str)
        + " "
        + paired_df["text2"].astype(str)
        + " "
        + paired_df["time_diff"].astype(str)
    )
    return paired_df[["user_id", "text_id", "text"]]


def _predict(
    texts: List[str],
    model_dir: str,
    batch_size: int = 32,
    max_length: int = 512,
    device: str | None = None,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
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
            if logits.ndim == 0:  # single item
                logits = np.array([float(logits)], dtype=np.float32)
            preds_all.extend(logits.tolist())

    return np.asarray([round(n, 0) for n in preds_all], dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a fine-tuned RoBERTa-base state-change regression model and write predictions back to CSV."
    )
    parser.add_argument("--csv", required=True, help="Path to input CSV (same schema as training).")
    parser.add_argument("--model_dir", required=True, help="Directory that contains the fine-tuned model and tokenizer.")
    parser.add_argument("--target",
                        choices=["state_change_valence", "state_change_arousal"],
                        required=True,
                        help="Which target this model predicts; determines destination column in the CSV.")
    parser.add_argument("--feature",
                        choices=["valence", "arousal"],
                        required=True,
                        help="Feature used to build pairs (must match what the model was trained with).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--max_length", type=int, default=512, help="Tokenizer max length.")
    parser.add_argument("--output_csv", default=None,
                        help="Optional output CSV path. If not provided, overwrites the input CSV safely by writing alongside with a suffix.")
    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.csv)

    # Validate columns present (must match training data schema)
    _ensure_cols(
        df,
        [
            "user_id",
            "text_id",
            "text",
            "timestamp",
            args.feature,  # 'valence' or 'arousal' used to create 'state1'
        ],
    )

    # Build paired dataset and model inputs
    paired_df = _prepare_inference_dataframe(df, feature_col=args.feature)
    if paired_df.empty:
        # If no pairs (e.g., all users have single record), nothing to predict; create an empty column if missing and save.
        if args.target not in df.columns:
            df[args.target] = np.nan
        out_path = args.output_csv or _default_out_path(args.csv, args.target)
        df.to_csv(out_path, index=False)
        print(f"No pairs to predict. Wrote CSV to {out_path}")
        return

    # Predict
    preds = _predict(
        texts=paired_df["text"].tolist(),
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Attach predictions to their "current" text_id rows in the original CSV
    pred_map = dict(zip(paired_df["text_id"].tolist(), preds.tolist()))

    # Ensure destination column exists
    if args.target not in df.columns:
        df[args.target] = np.nan

    # Fill predictions only for rows that were the "second" element in a pair (current row)
    # Keep existing ground-truth values if present; only fill NaNs
    mask_current_rows = df["text_id"].isin(pred_map.keys())
    to_fill_idx = df.index[mask_current_rows & df[args.target].isna()]
    df.loc[to_fill_idx, args.target] = df.loc[to_fill_idx, "text_id"].map(pred_map)

    out_path = args.output_csv or _default_out_path(args.csv, args.target)
    df.to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}")


def _default_out_path(input_csv: str, target: str) -> str:
    base, ext = os.path.splitext(input_csv)
    return f"{base}.{target}.predicted{ext}"


if __name__ == "__main__":
    main()