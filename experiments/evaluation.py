import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, f1_score
from scipy.stats import pearsonr


class MetricComputer:
    """
    Computes regression and classification metrics without using `evaluate`.
    """

    def compute(self, eval_pred):
        preds, labels = eval_pred

        # Ensure numpy arrays
        preds = np.squeeze(np.array(preds, dtype=float))
        labels = np.squeeze(np.array(labels, dtype=float))

        # Round predictions to nearest integer for classification metrics
        preds_rounded = np.round(preds).astype(int)
        labels_int = labels.astype(int)

        mse_val = mean_squared_error(labels, preds)
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(labels, preds)
        mae_val = mean_absolute_error(labels, preds)
        pearson_val = pearsonr(labels, preds)[0]
        f1_val = f1_score(labels_int, preds_rounded, average="weighted")

        return {
            "mse": round(mse_val, 2),
            "rmse": round(rmse_val, 2),
            "mae": round(mae_val, 2),
            "r_squared": round(r2_val, 2),
            "pearson": round(pearson_val, 2),
            "f1": round(f1_val, 2)
        }


# -------------------------------
# Main
# -------------------------------
def main(args):
    metric = MetricComputer()
    for i in range(1, 6):
        file_path = f"{args.base_dir}/{args.file_template.format(n=i)}"
        print(f"Processing file: {file_path}")

        task1_data = pd.read_csv(file_path)

        task1_data['pred_valence'] = task1_data['pred_valence'].fillna(0)
        task1_data['pred_arousal'] = task1_data['pred_arousal'].fillna(1)

        # Compute valence metric
        labels_valence = np.array(task1_data['true_valence'])
        preds_valence = np.array(task1_data['pred_valence'])
        valence_metric = metric.compute((preds_valence, labels_valence))
        print("Valence:", valence_metric)

        # Compute arousal metric
        labels_arousal = np.array(task1_data['true_arousal'])
        preds_arousal = np.array(task1_data['pred_arousal'])
        arousal_metric = metric.compute((preds_arousal, labels_arousal))
        print("Arousal:", arousal_metric)
        print("="*20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute metrics for task 1")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing CSV files")
    parser.add_argument("--file_template", type=str, required=True, help="Filename template with {n} placeholder")
    args = parser.parse_args()

    main(args)
