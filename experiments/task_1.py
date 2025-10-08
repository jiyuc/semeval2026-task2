import argparse
from argparse import ArgumentParser
import re
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def load_model(model_name: str):
    """Load the LLM model and tokenizer for translation."""
    print(f"Loading translation model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator, model, tokenizer

def prediction_task1(generator, tokenizer, text: str) -> str:
    prompt = f"""
    You are a helpful assistant that predicts emotional valence and arousal of a text.
    - Valence ranges from -2 (very negative) to 2 (very positive)
    - Arousal ranges from 0 (low) to 2 (high)
    Given the text: "{text}"
    Return your prediction as a JSON: {{ "valence": <number>, "arousal": <number> }}
    """

    output = generator(
        prompt,
        max_new_tokens=200,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )
    response = output[0]["generated_text"]
    valence_match = re.search(r'"valence":\s*([-+]?\d*\.?\d+)', response)
    arousal_match = re.search(r'"arousal":\s*([-+]?\d*\.?\d+)', response)
    prediction = {
        "valence": float(valence_match.group(1)) if valence_match else None,
        "arousal": float(arousal_match.group(1)) if arousal_match else None
    }
    return prediction


def evaluation(file_name):
    data = pd.read_csv(file_name)

    for item in ['valence', 'arousal']:
        # Flatten both columns into a single array for dataset-level evaluation
        y_true_all = data[[f'true_{item}']].values.flatten()
        y_pred_all = data[[f'pred_{item}']].values.flatten()

        # Calculate dataset-level metrics
        mse = mean_squared_error(y_true_all, y_pred_all)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_all, y_pred_all)
        r2 = r2_score(y_true_all, y_pred_all)

        print(f"Dataset-level metrics for {item}:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R2:   {r2:.4f}")


def main(args):
    print(f"Loading dataset {args.input_file}")
    df = pd.read_csv(args.input_file)
    # Load translation model
    generator, model, tokenizer = load_model(args.model_name)

    prediction = []

    # Prediction
    for index, row in df.iterrows():
        if index<50:
            text = row['text']
            pred = prediction_task1(generator, tokenizer, text)
            try:
                sample = {
                    'text': text,
                    'pred_valence': pred['valence'],
                    'pred_arousal': pred['arousal'],
                    'true_valence': row['valence'],   # add ground truth
                    'true_arousal': row['arousal']    # add ground truth
                }
            except:
                sample = {
                    'text': text,
                    'pred_valence': None,
                    'pred_arousal': None,
                    'true_valence': row['valence'],   # add ground truth
                    'true_arousal': row['arousal']    # add ground truth
            }
            prediction.append(sample)

    # Convert to DataFrame
    pred_df = pd.DataFrame(prediction)

    # Save to CSV
    pred_df.to_csv(args.output_file, index=False)
    print(f"Predictions with ground truth saved to {args.output_file}")

    # Evaluation
    print(evaluation(args.output_file))

if __name__ == "__main__":
    parser = ArgumentParser(description="Semeval shared task 2 subtask1")
    parser.add_argument("--input_file", type=str, default="TRAIN_RELEASE_3SEP2025/train_subtask1.csv")
    parser.add_argument("--model_name", type=str, default="gpt-oss")
    parser.add_argument("--output_file", type=str, default="TRAIN_RELEASE_3SEP2025/train_subtask1.csv")
    args = parser.parse_args()
    main(args)
