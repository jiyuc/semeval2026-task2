import argparse
import numpy as np
from typing import Dict, Any
from datasets import load_dataset, DatasetDict, Dataset
import wandb
import math
import os

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import evaluate
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

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
        return (time2 - time1).total_seconds() / (3600*24) # convert to hours

    def _exponential_decay(self, state_0, example1, example2, lambda_decay=0.5):
        """
        Calculate the exponential decay of a state value between two time points.

        Parameters:
        S_0 (float): The initial state value at time t1.
        t1 (float): The initial time point.
        t2 (float): The second time point.
        lambda_decay (float): The decay rate (λ).

        Returns:
        float: The state value at time t2.
        """
        # Calculate the state value at time t2 based on the decay formula
        delta_t = self._calculate_time_diff(example1, example2)
        state_1 = state_0 * np.exp(-lambda_decay * delta_t)

        return state_1

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
                    'time_diff': self._exponential_decay(example1[f'{self.feature}'], example1, example2),
                    # self._calculate_time_diff(example1, example2),
                    f'{self.label}': example2[self.label],  # predict the valence/arousal of the second sequence
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
            'text': f"state1: {str(example['state1'])} [SEP] content1: {example['text1']} [SEP] Time Elapse: {example['time_diff']} [SEP] content2: {example['text2']}"
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
        self._f1 = evaluate.load("f1")  # predictions are rounded to the closest integer, thus enable f1

    def compute(self, eval_pred):
        """
        Compute metrics given model predictions and labels.
        """
        preds, labels = eval_pred

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


class JointRobertaRegressor(PreTrainedModel):
    """
    Apply a joint modeling approach to RoBERTa for regression with temporal scaling.
    """
    def __init__(self, base_model_name: str):
        super().__init__(AutoConfig.from_pretrained(base_model_name))
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.backbone = AutoModelForSequenceClassification.from_pretrained(base_model_name,
                                                                    num_labels=self.config.hidden_size,  # Regression: only 1 output label
                                                                    problem_type="regression",  # This ensures it's set up for regression (not classification)
                                                                )
        self.hidden = self.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        # Define a simple Feedforward Network (FNN) for processing the concatenated embeddings
        self.regression = nn.Linear(self.hidden + 1, 1),  # [H + 1] because we concatenate scale with embeddings

    @staticmethod
    def _scaler(value: torch.Tensor):
        t = torch.clamp(value, min=0.0)
        s = 0.5 + (torch.log1p(t) / (torch.log1p(t + 1.0) + 1e-6)) * 1.5
        return torch.clamp(s, 0.5, 2.0)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        time_diff=None,
        labels=None,
    ):
        assert input_ids is not None, "Both sequences are required."
        device = input_ids.device

        # 1) Get encoder outputs from RoBERTa model
        encoder_outputs = self.backbone.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Unpack the shape of `last_hidden_state` into B, L, and H
        B, L, H = encoder_outputs.last_hidden_state.shape  # [B, L, H]

        # 2) Handle time_diff and apply scaling
        if time_diff is None:
            time_diff = torch.zeros(B, device=device)
        else:
            time_diff = time_diff.to(device).view(B)

        # Apply the time-based scaling
        scale = time_diff.view(B, 1, 1).squeeze(-1)  # [B, 1]
        # print(scale)

        # 3) Get last hidden state from the encoder output
        last_hidden_state = encoder_outputs.last_hidden_state  # [B, L, H]

        # 4) Concatenate the scale with the last hidden state for each token
        scale_expanded = scale.unsqueeze(1).expand(-1, L, -1)  # [B, L, 1] - repeat for each token
        concatenated = torch.cat((last_hidden_state, scale_expanded), dim=-1)  # [B, L, H + 1]

        # 5) Pass the concatenated embedding through the Feedforward Network (FNN)
        _concatenated = self.dropout(concatenated)
        logits = self.regression(_concatenated).mean(dim=1).squeeze(-1)  # [B, L, 1] - token-wise output for each token


        # 6) Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(logits.view(-1), labels.view(-1).float())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )



class RegressionTrainer:
    """
    Orchestrates tokenizer/model creation and fine-tuning for sequence regression with dual-sequence RoPE input.
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
            run_name: str = "default",
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = JointRobertaRegressor(model_name)
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
            run_name=run_name,
        )
        self._trainer = None

    def build_trainer(self, encoded: DatasetDict, metrics: MetricComputer) -> Trainer:
        """
        Create a Hugging Face Trainer for the regression task with custom collator for dual sequences.
        """
        def data_collator(features):
            batch = {}
            keys = ["input_ids", "attention_mask"]
            for k in keys:
                batch[k] = torch.tensor([f[k] for f in features], dtype=torch.long)
            batch["time_diff"] = torch.tensor([f.get("time_diff", 0.0) for f in features], dtype=torch.float16)
            if "labels" in features[0]:
                batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.float16)
            return batch

        # # Freeze all layers of the model (except the classification head)
        # for param in self.model.roberta.parameters():
        #     param.requires_grad = True
        #
        # # Only the classifier head will remain trainable
        # for param in self.model.classifier.parameters():
        #     param.requires_grad = True

        self._trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=encoded["train"],
            eval_dataset=encoded["validation"],
            compute_metrics=metrics.compute,
            data_collator=data_collator,
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
    CLI entry point to fine-tune a regression model with dual-sequence RoPE encoding.
    """
    parser = argparse.ArgumentParser(description="Fine-tune state-change regression with dual-sequence RoPE-encoded inputs.")
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
    parser.add_argument("--run_name", type=str, default="default", help="Name of the run.")
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
        run_name=args.run_name,
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

    # # Save final model if requested
    # reg.save(args.save_dir)
    # print(f"Final model and tokenizer saved in {args.save_dir}")


if __name__ == "__main__":
    main()