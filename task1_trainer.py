import argparse
import numpy as np
from typing import Dict, Tuple, Any
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate


class DataPreprocessor:
    """
    End-to-end dataset preparation:
    - load CSV splits
    - normalize column names
    - clean target column
    - scale labels consistently
    - tokenize and build label column
    """

    def __init__(
        self,
        data_files: Dict[str, str],
        label: str = "valence",
        feature_range: Tuple[float, float] = (-2, 2),
        text_col_candidates=("text", "Tweet", "text_body"),
        label_col_candidates=("valence", "Intensity Score", "arousal"),
    ):
        """
        Initialize the preprocessor with file mappings and column preferences.

        Args:
            data_files: Mapping of split name to CSV path (e.g., {"train": "...", "validation": "..."}).
            label: Target column to regress on (e.g., "valence").
            feature_range: MinMax scaling range applied using train split statistics.
            text_col_candidates: Possible text column names to normalize to "text".
            label_col_candidates: Possible label column names to normalize to the given feature name.
        """
        self.data_files = data_files
        self.label = label
        self.feature_range = feature_range
        self.text_col_candidates = text_col_candidates
        self.label_col_candidates = label_col_candidates
        self._scaler_params = None  # (vmin, vmax, scale, min_adj)

    def load(self) -> DatasetDict:
        """
        Load dataset from CSV files into a DatasetDict.

        Returns:
            A DatasetDict with 'train'/'validation'/optional 'test' splits.
        """
        return load_dataset("csv", data_files=self.data_files)

    def _normalize_columns(self, ds: DatasetDict) -> DatasetDict:
        """
        Normalize text and label column names to 'text' and self.feature.

        Returns:
            DatasetDict with unified column names.
        """
        # Normalize text column
        train_cols = ds["train"].column_names
        if "text" not in train_cols:
            for cand in self.text_col_candidates:
                if cand in train_cols:
                    ds = ds.rename_column(cand, "text")
                    break

        # Normalize label column
        train_cols = ds["train"].column_names
        if self.label not in train_cols:
            for cand in self.label_col_candidates:
                if cand in train_cols:
                    ds = ds.rename_column(cand, self.label)
                    break
        return ds

    def _clean_target(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert target to float and set invalid to None so downstream filter can drop them.

        Returns:
            Dict with cleaned target value or None.
        """
        v = example.get(self.label)
        try:
            return {self.label: float(v)}
        except (TypeError, ValueError):
            return {self.label: None}

    def _has_target(self, example: Dict[str, Any]) -> bool:
        """
        Predicate for filtering out rows with missing targets.

        Returns:
            True if target exists, else False.
        """
        return example[self.label] is not None

    @staticmethod
    def _fit_minmax_on_split(ds_split, col: str, feature_range: Tuple[float, float]) -> Tuple[float, float, float, float]:
        """
        Compute MinMax parameters on a split.

        Returns:
            (vmin, vmax, scale, min_adj) tuple to apply x*scale + min_adj.
        """
        arr = np.asarray(ds_split[col], dtype=np.float32)
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        if vmax == vmin:
            scale = 0.0
            min_adj = feature_range[0]
        else:
            scale = (feature_range[1] - feature_range[0]) / (vmax - vmin)
            min_adj = feature_range[0] - vmin * scale
        return vmin, vmax, scale, min_adj

    @staticmethod
    def _apply_minmax(batch: Dict[str, Any], col: str, scale: float, min_adj: float) -> Dict[str, Any]:
        """
        Apply precomputed MinMax scaling to a batch.

        Returns:
            Dict with scaled values under the original column name.
        """
        vals = np.asarray(batch[col], dtype=np.float32)
        if scale == 0.0:
            scaled = np.full_like(vals, fill_value=min_adj, dtype=np.float32)
        else:
            scaled = vals * scale + min_adj
        return {col: scaled.tolist()}

    @staticmethod
    def _to_labels(batch: Dict[str, Any], from_col: str) -> Dict[str, Any]:
        """
        Create 'labels' column from a numeric source column.

        Returns:
            Dict with labels as float32.
        """
        vals = np.array(batch[from_col], dtype=np.float32)
        return {"labels": vals}

    def prepare(
        self,
        tokenizer,
        max_length: int = 512,
    ) -> DatasetDict:
        """
        Execute full preprocessing:
        - load
        - normalize columns
        - clean and filter targets
        - scale targets using train stats
        - tokenize
        - add labels and trim columns

        Args:
            tokenizer: A Hugging Face tokenizer instance.
            max_length: Max sequence length for tokenization.

        Returns:
            Encoded and ready-to-train DatasetDict.
        """
        dataset = self.load()
        dataset = self._normalize_columns(dataset)

        dataset = dataset.map(self._clean_target)
        dataset = dataset.filter(self._has_target)

        # Fit scaler on train split
        vmin, vmax, scale, min_adj = self._fit_minmax_on_split(
            dataset["train"], col=self.label, feature_range=self.feature_range
        )
        self._scaler_params = (vmin, vmax, scale, min_adj)

        # Apply scaling to all available splits consistently
        for split in dataset.keys():
            print(f"Before scaling {self.label} (train): min={min(dataset[split][self.label]):.4f} max={max(dataset[split][self.label]):.4f}")

            dataset[split] = dataset[split].map(
                self._apply_minmax,
                batched=True,
                fn_kwargs={"col": self.label, "scale": scale, "min_adj": min_adj},
            )

            print(f"After scaling {self.label} (train): min={min(dataset[split][self.label]):.4f} max={max(dataset[split][self.label]):.4f}")

        # Tokenization
        def _preprocess(batch):
            return tokenizer(
                batch["text"], truncation=True, padding="max_length", max_length=max_length
            )

        encoded = dataset.map(_preprocess, batched=True)

        # Labels
        encoded = encoded.map(self._to_labels, batched=True, fn_kwargs={"from_col": self.label})

        # Keep only necessary columns
        keep = {"input_ids", "attention_mask", "labels", "text", self.label}
        for split in encoded.keys():
            drop_cols = [c for c in encoded[split].column_names if c not in keep]
            encoded[split] = encoded[split].remove_columns(drop_cols)

        return encoded

    def scaler_info(self) -> Tuple[float, float, float, float]:
        """
        Get fitted scaler parameters (vmin, vmax, scale, min_adj).

        Returns:
            Tuple of MinMax parameters computed on the train split.
        """
        if self._scaler_params is None:
            raise RuntimeError("Scaler parameters are not available before calling prepare().")
        return self._scaler_params


class MetricComputer:
    """
    Wraps evaluation metrics for the Trainer.
    """

    def __init__(self):
        """
        Load required regression metrics from evaluate.
        """
        self._mse = evaluate.load("mse")
        self._pearson = evaluate.load("pearsonr")
        self._r2 = evaluate.load("r_squared")
        self._mae = evaluate.load("mae")
        self._rmse = evaluate.load("mse")
        self._f1 = evaluate.load("f1")  # the predictions are rounded to the closest categorical integer, thus, enable f1

    def compute(self, eval_pred):
        """
        Compute metrics given model predictions and labels.

        Args:
            eval_pred: Tuple(preds, labels) as provided by HF Trainer.

        Returns:
            Dict of scalar metrics.
        """
        preds, labels = eval_pred
        preds = np.squeeze(preds).astype(np.int8)
        preds = [round(n, 0) for n in preds]  # round to the closest integer
        labels = np.squeeze(labels).astype(np.int8)

        print(len(preds), len(labels))

        mse_val = float(self._mse.compute(predictions=preds, references=labels, squared=True)["mse"])
        pearson_val = float(self._pearson.compute(predictions=preds, references=labels)["pearsonr"])
        r2_val = float(self._r2.compute(predictions=preds, references=labels))
        mae_val = float(self._mae.compute(predictions=preds, references=labels)["mae"])
        rmse_val = float(self._rmse.compute(predictions=preds, references=labels, squared=False)["mse"])
        f1_val = float(self._f1.compute(predictions=preds, references=labels, average="weighted")["f1"])

        return {
            "mse": mse_val,
            "pearson": pearson_val,
            "r_squared": r2_val,
            "mae": mae_val,
            "rmse": rmse_val,
            "f1": f1_val,
        }


class RegressionTrainer:
    """
    Orchestrates tokenizer/model creation and fine-tuning for sequence regression.
    """

    def __init__(
        self,
        model_name: str,
        output_dir: str = "./results",
        learning_rate: float = 2e-5,
        train_bs: int = 16,
        eval_bs: int = 32,
        num_epochs: int = 5,
        weight_decay: float = 0.01,
        logging_dir: str = "./logs",
        report_to: str = "none",
        metric_for_best: str = "mse",
        greater_is_better: bool = False,
        save_total_limit: int = 2,
        eval_strategy: str = "epoch",
        save_strategy: str = "epoch",
    ):
        """
        Configure model, tokenizer, and training arguments.

        Args:
            model_name: Pretrained model identifier or local path.
            output_dir: Directory for checkpoints and outputs.
            learning_rate: Optimizer learning rate.
            train_bs: Per-device train batch size.
            eval_bs: Per-device evaluation batch size.
            num_epochs: Number of training epochs.
            weight_decay: Weight decay.
            logging_dir: Directory for logs.
            report_to: Reporting integrations ("none", "tensorboard", "wandb", etc.).
            metric_for_best: Metric to select best checkpoint.
            greater_is_better: Whether higher metric is better.
            save_total_limit: Max number of checkpoints to keep.
            eval_strategy: Evaluation frequency ("no", "steps", "epoch").
            save_strategy: Saving frequency ("no", "steps", "epoch").
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            problem_type="regression",
        )
        self.args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            learning_rate=learning_rate,
            per_device_train_batch_size=train_bs,
            per_device_eval_batch_size=eval_bs,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model=metric_for_best,
            greater_is_better=greater_is_better,
            save_total_limit=save_total_limit,
            logging_dir=logging_dir,
            report_to=report_to,
        )
        self._trainer = None

    def build_trainer(self, encoded: DatasetDict, metrics: MetricComputer) -> Trainer:
        """
        Create a Hugging Face Trainer for the regression task.

        Args:
            encoded: Tokenized and labeled DatasetDict.
            metrics: MetricComputer instance.

        Returns:
            Configured Trainer.
        """
        self._trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=encoded["train"],
            eval_dataset=encoded["validation"],
            compute_metrics=metrics.compute,
        )
        return self._trainer

    def fit(self):
        """
        Run fine-tuning using the built Trainer.
        """
        if self._trainer is None:
            raise RuntimeError("Trainer is not built. Call build_trainer() first.")
        self._trainer.train()

    def evaluate(self, dataset_split) -> Dict[str, float]:
        """
        Evaluate current model on a dataset split.

        Args:
            dataset_split: A Dataset (e.g., encoded['validation']) to evaluate on.

        Returns:
            Dict of evaluation metrics.
        """
        if self._trainer is None:
            raise RuntimeError("Trainer is not built. Call build_trainer() first.")
        return self._trainer.evaluate(dataset_split)

    def save(self, path: str):
        """
        Save the fine-tuned model and tokenizer.

        Args:
            path: Target directory to save artifacts.
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def main():
    """
    CLI entry point to fine-tune a regression model.

    Example:
        python task1_trainer.py \
            --train data/SemEval2026/data/split/subtask1_train.csv \
            --validation data/SemEval2026/data/split/subtask1_test.csv \
            --model cardiffnlp/twitter-xlm-roberta-base \
            --output_dir ./results \
            --save_dir ./xlm_valence_reg_model
    """
    parser = argparse.ArgumentParser(description="Fine-tune a regression model on valence/arousal data.")
    parser.add_argument("--train", required=True, help="Path to training CSV.")
    parser.add_argument("--validation", required=True, help="Path to validation CSV.")
    parser.add_argument("--test", help="Path to test CSV (optional).")
    parser.add_argument("--model", default="cardiffnlp/twitter-xlm-roberta-base", help="HF model name or local path.")
    parser.add_argument("--output_dir", default="./results", help="Trainer output directory.")
    parser.add_argument("--save_dir", default="./xlm_valence_reg_model", help="Directory to save final model.")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--train_bs", type=int, default=16, help="Per-device train batch size.")
    parser.add_argument("--eval_bs", type=int, default=32, help="Per-device eval batch size.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--report_to", default="none", help="Reporting integration for Trainer.")
    parser.add_argument("--label", type=str, default="valence", help="The target label name.")
    parser.add_argument("--scale_min", type=float, default=-2, help="Min of scaling range.")
    parser.add_argument("--scale_max", type=float, default=2, help="Max of scaling range.")
    args = parser.parse_args()

    data_files = {"train": args.train, "validation": args.validation}
    if args.test:
        data_files["test"] = args.test

    # Build components
    reg = RegressionTrainer(
        model_name=args.model,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        train_bs=args.train_bs,
        eval_bs=args.eval_bs,
        num_epochs=args.epochs,
        weight_decay=args.weight_decay,
        report_to=args.report_to,

    )
    pre = DataPreprocessor(
        data_files=data_files,
        label=args.label,
        feature_range=(args.scale_min, args.scale_max),
    )
    metrics = MetricComputer()

    # Prepare data
    encoded = pre.prepare(tokenizer=reg.tokenizer, max_length=args.max_length)

    # Build, train, evaluate
    trainer = reg.build_trainer(encoded, metrics)
    reg.fit()

    # Prefer test if provided; else evaluate on validation
    eval_split_name = "test" if "test" in encoded else "validation"
    results = reg.evaluate(encoded[eval_split_name])
    print(f"{eval_split_name} results:", results)

    # Save artifacts
    reg.save(args.save_dir)
    print(f"Final model and tokenizer saved in {args.save_dir}")


if __name__ == "__main__":
    main()