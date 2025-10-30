import argparse
import numpy as np
from typing import Dict, Any
from datasets import load_dataset, DatasetDict, Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
from datetime import datetime
import pandas as pd
import os

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="semeval2026"

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="end"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"


class DataPreprocessor:
    def __init__(
            self,
            data_files: Dict[str, str],
            label: str = "",
            feature: str = "",
            text_col_candidates=("text", "Tweet", "text_body"),
            label_col_candidates=("valence", "Intensity Score", "arousal"),

    ):
        self.data_files = data_files
        self.label = label
        self.feature = feature
        self.text_col_candidates = text_col_candidates
        self.label_col_candidates = label_col_candidates


    def load(self) -> DatasetDict:
        """
        Load dataset from CSV files into a DatasetDict.
        """
        return load_dataset("csv", data_files=self.data_files)

    def _normalize_columns(self, ds: DatasetDict) -> DatasetDict:
        """
        Normalize text and label column names to 'text' and self.feature.
        """
        train_cols = ds["train"].column_names
        if "text" not in train_cols:
            for cand in self.text_col_candidates:
                if cand in train_cols:
                    ds = ds.rename_column(cand, "text")
                    break

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
        """
        v = example.get(self.label)
        try:
            return {self.label: float(v)}
        except (TypeError, ValueError):
            return {self.label: None}

    def _has_target(self, example: Dict[str, Any]) -> bool:
        """
        Predicate for filtering out rows with missing targets.
        """
        return example[self.label] is not None

    @staticmethod
    def _to_labels(batch: Dict[str, Any], from_col: str) -> Dict[str, Any]:
        """
        Create 'labels' column from a numeric source column.
        """
        vals = np.array(batch[from_col], dtype=np.float32)
        return {"labels": vals}

    def _calculate_time_diff(self, example1, example2) -> float:
        """
        Calculate the time difference between two consecutive examples.
        """
        time_format = "%Y-%m-%d %H:%M:%S"  # Assumes the timestamp is in the format of 'YYYY-MM-DD HH:MM:SS'
        time1 = datetime.strptime(example1['timestamp'], time_format)
        time2 = datetime.strptime(example2['timestamp'], time_format)
        return (time2 - time1).total_seconds()

    def _pair_sequences_by_user_id(self, dataset: Dataset) -> Dataset:
        """
        Pair consecutive sequences for each user and add features.
        Keep the last sequence but remove the `None` label (set it to `0`).
        """
        user_pairs = []
        df = pd.DataFrame(dataset)
        user_groups = df.groupby("user_id")

        for user_id, group in user_groups:
            group = group.sort_values(by=["timestamp"])

            # Iterate over the sequences, but stop before the last one
            for i in range(1, len(group) - 1):  # Exclude the last sequence
                example1 = group.iloc[i - 1]
                example2 = group.iloc[i]

                # Create a new example for the pair with a state_change value
                user_pairs.append({
                    'user_id': user_id,
                    'text1': example1['text'],
                    'text2': example2['text'],
                    'state1': example1[f'{self.feature}'],
                    'time_diff': self._calculate_time_diff(example1, example2),
                    f'{self.label}': example1[self.label],
                    'timestamp1': example1['timestamp'],
                    'timestamp2': example2['timestamp'],
                })


        # Convert the processed user pairs back to a Hugging Face Dataset
        processed_df = pd.DataFrame(user_pairs)
        processed_dataset = Dataset.from_pandas(processed_df)

        return processed_dataset

    @staticmethod
    def combine_text(example):
        """
        Concatenate text1, state1, and text2 to create a new 'text' column.
        The columns are joined in the order: text1, state1, text2.
        """
        return {
            'text': f"state1: {str(example['state1'])} [SEP] content1: {example['text1']} [SEP] content2: {example['text2']} [SEP] time: {example['time_diff']}"
            # 'text': f"state: {str(example['state1'])} [SEP] content: {example['text1']} [SEP] content: {example['text2']}"
        }

    def _create_text_column(self, paired_dataset):
        """
        Apply the `combine_text` function to the dataset to create a new 'text' column.
        """
        return paired_dataset.map(self.combine_text, batched=False)

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
        - add labels and trim columns for all splits (train, validation, test)
        """
        # Load dataset
        dataset = self.load()

        # Normalize column names across all splits
        dataset = self._normalize_columns(dataset)

        # Apply the target cleaning and filtering for each split
        dataset = dataset.map(self._clean_target)
        dataset = dataset.filter(self._has_target)

        # Initialize a new DatasetDict to store the processed splits
        processed_dataset = DatasetDict()

        # Process each split (train, validation, and test)
        for split_name in dataset.keys():
            print(f"Processing split: {split_name}")

            # Pair sequences by user_id and add features for the current split
            paired_split = self._pair_sequences_by_user_id(dataset[split_name])
            paired_split = self._create_text_column(paired_split)

            # Tokenization step
            def _preprocess(batch):
                return tokenizer(
                    batch['text'], truncation=True, padding="max_length", max_length=max_length
                )

            # Tokenize the paired dataset
            encoded = paired_split.map(_preprocess, batched=True)

            # Convert the 'state_change_valence or state_change_arousal' to 'labels'
            encoded = encoded.map(self._to_labels, batched=True, fn_kwargs={"from_col": self.label})

            # Add the processed split to the new DatasetDict
            processed_dataset[split_name] = encoded

        # Return the processed dataset
        return processed_dataset


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
        # self._f1 = evaluate.load("f1")  # the predictions are rounded to the closest categorical integer, thus, enable f1

    def compute(self, eval_pred):
        """
        Compute metrics given model predictions and labels.

        Args:
            eval_pred: Tuple(preds, labels) as provided by HF Trainer.

        Returns:
            Dict of scalar metrics.
        """
        preds, labels = eval_pred
        # preds = np.squeeze(preds).astype(np.int8)
        # preds = [round(n, 0) for n in preds]  # round to the closest integer
        # labels = np.squeeze(labels).astype(np.int8)

        # print(len(preds), len(labels))

        mse_val = float(self._mse.compute(predictions=preds, references=labels, squared=True)["mse"])
        pearson_val = float(self._pearson.compute(predictions=preds, references=labels)["pearsonr"])
        r2_val = float(self._r2.compute(predictions=preds, references=labels))
        mae_val = float(self._mae.compute(predictions=preds, references=labels)["mae"])
        rmse_val = float(self._rmse.compute(predictions=preds, references=labels, squared=False)["mse"])
        # f1_val = float(self._f1.compute(predictions=[round(n, 0) for n in np.squeeze(preds)],
        #                                 references=np.squeeze(labels).astype(np.int8), average="weighted")["f1"])

        return {
            "mse": mse_val,
            "pearson": pearson_val,
            "r_squared": r2_val,
            "mae": mae_val,
            "rmse": rmse_val,
            # "f1": f1_val,
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
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load the model with regression head (num_labels=1)
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
        """
        if self._trainer is None:
            raise RuntimeError("Trainer is not built. Call build_trainer() first.")
        return self._trainer.evaluate(dataset_split)

    def save(self, path: str):
        """
        Save the fine-tuned model and tokenizer.
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def main():
    """
    CLI entry point to fine-tune a regression model.
    """
    parser = argparse.ArgumentParser(description="Fine-tune a regression model on valence/arousal data.")
    parser.add_argument("--train", type=str, required=True, help="Path to the training dataset")
    parser.add_argument("--validation", type=str, required=True, help="Path to the validation dataset")
    parser.add_argument("--test", type=str, default=None, help="Path to the test dataset")
    parser.add_argument("--model", type=str, default="cardiffnlp/twitter-xlm-roberta-base", help="Pretrained model name")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for results")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--train_bs", type=int, default=16, help="Batch size for training")
    parser.add_argument("--eval_bs", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--report_to", type=str, default="none", help="Reporting options for logging")
    parser.add_argument("--max_length", type=int, default=512, help="Max length for tokenization")
    parser.add_argument("--save_dir", type=str, default="./saved_model", help="Directory to save the model")
    parser.add_argument("--label", type=str, default="state_change_valence", help="The target label name.")
    parser.add_argument("--feature", type=str, default="valence", help="The valence or arousal feature.")

    args = parser.parse_args()

    data_files = {"train": args.train, "validation": args.validation}
    if args.test:
        data_files["test"] = args.test

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
        feature=args.feature,
    )
    metrics = MetricComputer()

    # Prepare data
    encoded = pre.prepare(tokenizer=reg.tokenizer, max_length=args.max_length)

    # Build, train, evaluate
    trainer = reg.build_trainer(encoded, metrics)
    reg.fit()

    eval_split_name = "test" if "test" in encoded else "validation"
    results = reg.evaluate(encoded[eval_split_name])
    print(f"{eval_split_name} results:", results)

    # reg.save(args.save_dir)
    # print(f"Final model and tokenizer saved in {args.save_dir}")

if __name__ == "__main__":
    main()