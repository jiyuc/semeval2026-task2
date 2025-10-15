# ... existing code ...
from tqdm import tqdm


def _ensure_cols(df: pd.DataFrame, col_names: List[str]) -> None:
    missing = [c for c in col_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
# ... existing code ...


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
# ... existing code ...


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

    return np.asarray([round(n) for n in preds_all], dtype=np.int8)  # round to int
# ... existing code ...


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a fine-tuned RoBERTa-base model and write rounded predictions back to CSV."
    )
    parser.add_argument("--csv", required=True, help="Path to input CSV (same schema as training).")
    parser.add_argument("--model_dir", required=True, help="Directory that contains the fine-tuned model and tokenizer.")
    parser.add_argument("--target",
                        choices=["valence", "arousal"],
                        required=True,
                        help="Which column to fill with predictions (rounded to int).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--max_length", type=int, default=512, help="Tokenizer max length.")
    parser.add_argument("--output_csv", default=None,
                        help="Optional output CSV path. If omitted, writes alongside input with a suffix.")
    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.csv)

    # Validate columns present (must match training data schema for task1-style inputs)
    _ensure_cols(
        df,
        [
            "text",
            "user_id",
            "text_id",
            "timestamp",
        ],
    )

    # Tokenize plain 'text' rows (no pairing for this inference; predicts current state)
    texts = df["text"].astype(str).tolist()
    if len(texts) == 0:
        if args.target not in df.columns:
            df[args.target] = np.nan
        out_path = args.output_csv or _default_out_path(args.csv, args.target)
        df.to_csv(out_path, index=False)
        print(f"No texts to predict. Wrote CSV to {out_path}")
        return

    preds = _predict(
        texts=texts,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Ensure destination column exists
    if args.target not in df.columns:
        df[args.target] = np.nan

    # Fill by index; keep any existing non-null values
    mask_to_fill = df[args.target].isna()
    df.loc[mask_to_fill, args.target] = np.asarray(preds, dtype=np.int8)[mask_to_fill.values]

    out_path = args.output_csv or _default_out_path(args.csv, args.target)
    df.to_csv(out_path, index=False)
    print(f"Predictions (rounded) saved to {out_path}")
# ... existing code ...