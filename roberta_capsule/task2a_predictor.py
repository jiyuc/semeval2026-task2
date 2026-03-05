import argparse
import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from safetensors.torch import load_file
# Import the model and preprocessing logic from the trainer
from task2a_trainer import (
    RobertaForSequenceRegression,
    DataPreprocessor,
    CustomDataCollator
)


def load_and_preprocess(input_csv, tokenizer, label_col, feature_col, history_N, max_length):
    # Reuse the trainer's preprocessor logic to ensure features match training
    preprocessor = DataPreprocessor(
        data_files={"test": input_csv},
        label=label_col,
        feature=feature_col
    )

    # Load raw data
    raw_dataset = preprocessor.load()["test"]
    raw_dataset = raw_dataset.filter(lambda x: x['is_forecasting_user'] == True)

    # Generate phase samples (this handles the user-id indexing logic)
    # Pass split_name="test" to ensure it only takes the last phase if required,
    # or modify the trainer's logic if you need predictions for all historical points.
    paired_dataset = preprocessor._generate_collection_phase_sample_by_user_id(
        raw_dataset, split_name="test", history_N=history_N
    )
    paired_dataset = preprocessor._create_text_column(paired_dataset)


    # Tokenize
    def _tokenize(batch):
        return tokenizer(
            batch['text'], truncation=True, padding="max_length", max_length=max_length
        )

    encoded_dataset = paired_dataset.map(_tokenize, batched=True)
    return encoded_dataset, paired_dataset.to_pandas()


def predict(model, dataset, tokenizer, batch_size):
    collator = CustomDataCollator(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            # Move all tensors in batch to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            # Labels aren't needed for inference
            if "labels" in inputs:
                inputs.pop("labels")

            outputs = model(**inputs)
            logits = outputs.logits.view(-1).cpu().numpy()
            all_preds.extend(logits)

    return all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True, help="Directory containing the saved model")
    parser.add_argument('--base_model', type=str, default="cardiffnlp/twitter-xlm-roberta-base")
    parser.add_argument('--at_N', type=int, default=2)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--label', type=str, default="state_change_valence")
    # parser.add_argument('--feature', type=str, default="valence")
    args = parser.parse_args()

    meta_df = pd.DataFrame()
    for suffix in ["valence", "arousal"]:
        print(f"Loading model from {args.model_dir.format(suffix)}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir.format(suffix))
        # We use the custom class from task2a_trainer
        model = RobertaForSequenceRegression(args.base_model, history_N=args.at_N)


        # Load weights
        state_dict_path = os.path.join(args.model_dir.format(suffix), "pytorch_model.bin")
        if not os.path.exists(state_dict_path):
            state_dict = load_file(os.path.join(args.model_dir.format(suffix), "model.safetensors"))
        else:
            state_dict = torch.load(state_dict_path, map_location="cpu")

        model.load_state_dict(state_dict)

        print("Preprocessing input data...")
        encoded_ds, template = load_and_preprocess(
            args.input_csv, tokenizer, f'state_change_{suffix}', suffix, args.at_N, args.max_length
        )
        try:
            meta_df = meta_df.merge(
                template,
                on="user_id",
                how="left"
            )
        except KeyError:
            print(f"Template does not contain 'user_id' column. Apply concatenation instead.")
            meta_df = pd.concat([meta_df, template], axis=1)

        print("Running inference...")
        preds = predict(model, encoded_ds, tokenizer, args.batch_size)

        # Combine metadata (user_id, collection_phase) with predictions
        meta_df[f"pred_state_change_{suffix}"] = [round(float(p), 2) for p in preds]

    meta_df = meta_df[["user_id","pred_state_change_valence","pred_state_change_arousal"]].copy()
    meta_df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    main()