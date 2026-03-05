from datasets import Dataset
from transformers import pipeline
import pandas as pd
import datasets
import argparse
from task1_trainer import RoBERTaWithRegressionHead
import os
import torch


# load dataset
def load_dataset(path: str) -> Dataset:
    df = pd.read_csv(path)
    # extract text, user_id, text_id column and convert to Dataset
    df['text'] = df['text'].map(lambda x: x.lower())
    return datasets.Dataset.from_pandas(df)


def build_prediction_pipeline(model_path: str) -> pipeline:
    assert model_path is not None, "Model path cannot be None"

    # 1. Instantiate using the original backbone string so it loads the 3-label config
    # Replace with the actual base model ID you used for finetuning
    base_id = 'cardiffnlp/twitter-roberta-base-sentiment-latest'

    model = RoBERTaWithRegressionHead(
        base_model_name=base_id,
        dropout_rate=0.0  # Set to 0 for inference
    )

    # 2. Load the finetuned weights (the backbone + your regression_head)
    state_dict_path = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.exists(state_dict_path):
        from safetensors.torch import load_file
        state_dict = load_file(os.path.join(model_path, "model.safetensors"))
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))

    model.eval()

    # 3. Use 'feature-extraction' to get the raw regression float from forward()
    return pipeline(
        task='feature-extraction',
        model=model,
        tokenizer=model_path,
        # device=0 if torch.cuda.is_available() else -1
    )


def predict(dataset: Dataset, blob: pipeline) -> list:
    assert dataset is not None, "Dataset cannot be None"
    assert blob is not None, "Pipeline cannot be None"

    texts = [str(t) for t in dataset['text']]
    raw_outputs = blob(texts, batch_size=32)

    # Extract the scalar value from the tensor result
    # In feature-extraction, the output for each text is typically [[[value]]]
    preds = []
    for out in raw_outputs:
        val = out[0]
        while isinstance(val, list):
            val = val[0]
        preds.append(float(val))
    return preds


def save_predictions(predictions: pd.DataFrame, output_path: str) -> None:
    assert predictions is not None, "Predictions cannot be None"
    assert output_path is not None, "Output path cannot be None"
    predictions = predictions[['user_id', 'text_id', 'pred_valence', 'pred_arousal']].copy()
    predictions.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input csv file')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to output csv file')
    parser.add_argument('--model_dir', type=str, default='./model/', required=True, help='Home directory of the models to use for prediction')
    parser.add_argument('--pred_col', type=str, default='pred_', required=True, help='Name of the column to store predictions in')
    args = parser.parse_args()
    data = load_dataset(args.input_csv)
    df = data.to_pandas()

    for suffix in ['valence', 'arousal']:
        raw_preds = predict(data, build_prediction_pipeline(args.model_dir+f'twitter-roberta-base-{suffix}-latest'))
        df[f"{args.pred_col}{suffix}"] = [round(float(p), 1) for p in raw_preds]

    # save predictions
    save_predictions(df, args.output_csv)

if __name__ == '__main__':
    main()
