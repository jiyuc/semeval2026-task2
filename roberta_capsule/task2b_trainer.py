import argparse
from pyexpat import features

import numpy as np
from typing import Dict, Any
from datasets import load_dataset, DatasetDict, Dataset
from pandas import Timedelta
import wandb
import math
import os
os.environ["TOKENIZERS_PARALLELISM"]="false"

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoModel, RobertaConfig,
)
import evaluate
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import MSELoss
import warnings

from transformers import AutoConfig, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


# os.environ['WANDB_API_KEY'] = '' # specify you wandb here if you want to use wandb to track the MLOps

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="semeval2026-subtask2a"

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="end"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"


class TimeSeries:
    def __init__(self, series, **kwargs):
        self.series = series
        self.kwargs = kwargs

    def calculate_mad(self, **kwargs):

        if len(self.series) >= 3:
            return self.series.mad()
        return 0

    def calculate_cv(self, **kwargs):
        # Calculates coefficient of variation or 0 if mean zero
        if len(self.series) >= 5:
            return self.series.std() / self.series.mean() if self.series.mean() != 0 else 0
        return 0

    def calculate_rolling_volatility(self, **kwargs):

        window_size = kwargs.get('window_size', 5)  # default to window size of 5
        if self.series is not None:
            return self.series.rolling(window=window_size).std().fillna(0)
        return pd.Series([0])

    def calculate_autocorr(self, **kwargs):
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")  # Ensure that all warnings are caught

                lag = kwargs.get('lag', 1)
                # Attempt to calculate autocorrelation
                result = self.series.autocorr(lag=lag)

                # Check if there are any warnings captured during autocorrelation calculation
                if caught_warnings:
                    for warning in caught_warnings:
                        continue
                        # print(f"Warning: {warning.message}")
                        # print(f"Problematic data: {self.series}")


                # Return autocorrelation result, default to 0 if there's an issue
                return result if not any(np.isnan(result)) else 0  # Prevent NaN results
        except Exception as e:
            # print(f"Error calculating autocorrelation: {e}")
            # print(f"Problematic data: {self.series}")
            return 0


class DataPreprocessor:
    def __init__(
            self,
            data_files: Dict[str, str],
            label: str = "",
            # feature: str = "",
    ):
        self.data_files = data_files
        self.label = label
        # self.feature = feature


    def load(self) -> DatasetDict:
        """
        Load dataset from CSV files into a DatasetDict.
        """
        return load_dataset("csv", data_files=self.data_files)


    @staticmethod
    def _to_labels(batch: Dict[str, Any], from_col: str) -> Dict[str, Any]:
        """
        Create 'labels' column from a numeric source column.
        """
        vals = np.array(batch[from_col], dtype=np.float32)
        return {"labels": vals}

    def _calculate_time_diff(self, prev, curr) -> float:
        """
        Calculate the time difference between two consecutive examples.
        """
        time_format = "%Y-%m-%d %H:%M:%S"  # Assumes the timestamp is in the format of 'YYYY-MM-DD HH:MM:SS'
        curr = datetime.strptime(curr['timestamp'], time_format)
        prev = datetime.strptime(prev['timestamp'], time_format)
        return (curr - prev).total_seconds() / (3600*24) # convert to hours

    def pad_features(self, values, length):
        """
        Reverse a list, pad with zeros, and reshape into fixed-length chunks.

        Parameters
        ----------
        values : list
            Input list of numeric values
        length : int
            Fixed chunk length

        Returns
        -------
        np.ndarray
            1D NumPy array with shape (length, )
        """
        if length == 0:
            return np.zeros(0)

        if len(values) == 0:
            return np.zeros(length)
        arr = np.asarray(values[:length])  # chunk
        pad_len = (-len(arr)) % length  # padding needed to reach multiple of length
        arr_padded = np.pad(arr, (0, pad_len), mode='constant')
        return arr_padded#.reshape(-1, length)


    def _generate_collection_phase_sample_by_user_id(self, dataset: Dataset, split_name: str, history_N: int) -> Dataset:
        """
        Pair consecutive sequences for each user and add features.
        Keep the last sequence but remove the `None` label (set it to `0`).
        """
        collection_group_samples = []
        df = pd.DataFrame(dataset)
        user_groups = df.groupby("user_id")  # group by each user


        for user_id, group in user_groups:
            first_half = group[group['group'] == 1] if split_name != 'test' else group
            second_half = group[group['group'] == 2] if split_name != 'test' else pd.DataFrame()

            collection_group_samples.append({
                'user_id': user_id,
                'text': first_half['text'].tolist()[-1],  # only get the last text in the first half
                'valence_mean': first_half['valence'].mean(),
                'arousal_mean': first_half['arousal'].mean(),
                'valences': self.pad_features(first_half['valence'].tolist()[::-1], length=history_N),
                'arousals': self.pad_features(first_half['arousal'].tolist()[::-1], length=history_N),
                'valence_mssd': self.time_adjusted_mssd(first_half.copy(), 'valence'),
                'arousal_mssd': self.time_adjusted_mssd(first_half.copy(), 'arousal'),
                'valence_autocorrelation': TimeSeries(first_half['valence']).calculate_autocorr(),
                'arousal_autocorrelation': TimeSeries(first_half['arousal']).calculate_autocorr(),
                'valence_rolling_volatility': self.pad_features(
                    TimeSeries(first_half['valence']).calculate_rolling_volatility(), length=history_N),
                'arousal_rolling_volatility': self.pad_features(
                    TimeSeries(first_half['arousal']).calculate_rolling_volatility(), length=history_N),
                'valence_swings': self.valence_swing_proportion(first_half.copy()),
                'bounce_to_inertia': self.bounce_to_inertia_ratio(first_half.copy()),
                f'disposition_change_{self.label}': round(second_half[self.label].mean() - first_half[self.label].mean(),2) if split_name != 'test' else 0
            }
            )

        # Convert the processed user pairs back to a Hugging Face Dataset
        processed_df = pd.DataFrame(collection_group_samples)
        print(f"Number of generated phase samples: {len(processed_df)}")
        processed_dataset = Dataset.from_pandas(processed_df)
        return processed_dataset


    @staticmethod
    def combine_text(example):
        return {
            'text': f"{str(example['text'])}"
        }

    @staticmethod
    def time_adjusted_mssd(user_data: pd.DataFrame, state_name: str) -> float:
        """Compute time-adjusted MSSD for a chronological sequence of posts.

        user_data must be sorted by 'create_time' and include a numeric 'valence' column.
        """
        if len(user_data) < 2:
            return 0.0

        user_data["timestamp"] = pd.to_datetime(user_data["timestamp"])
        # Calculate tau as the difference between max and min timestamps in seconds
        tau_seconds = (user_data["timestamp"].max() - user_data["timestamp"].min()).total_seconds()

        # If there's no variation in time (all timestamps are identical), return 0.0
        if tau_seconds == 0:
            return 0.0


        deltas = user_data["timestamp"].diff().dt.total_seconds().iloc[1:]
        diffs = user_data[state_name].diff().iloc[1:]

        # if no deltas or diffs (e.g., all 0.0), return 0.0
        if deltas.empty or diffs.empty:
            return 0.0

        w = np.exp(-deltas / tau_seconds)
        numerator = np.sum(w * (diffs ** 2))
        denominator = np.sum(w)
        mssd = numerator / denominator if denominator > 0 else 0.0
        return mssd

    @staticmethod
    def valence_swing_proportion(user_data: pd.DataFrame):
        signs = np.sign(user_data["valence"].fillna(0))
        sign_diff = signs.diff().fillna(0).abs()
        return float((sign_diff != 0).mean())

    @staticmethod
    def bounce_to_inertia_ratio(user_data: pd.DataFrame):
        signs = np.sign(user_data["valence"].fillna(0))
        prev = signs.shift(1)
        bounce = ((prev < 0) & (signs >= 0)).sum()
        total_negative = ((signs == -1) & (signs.shift(1) == -1)).sum()
        return float(bounce / (bounce + total_negative)) if (bounce + total_negative) != 0 else 0.0


    def _create_text_column(self, paired_dataset):
        """
        Apply the `combine_text` function to the dataset to create a new 'text' column.
        """
        return paired_dataset.map(self.combine_text, batched=False)

    def prepare(
            self,
            tokenizer,
            max_length: int,
            history_N: int
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

        # Initialize a new DatasetDict to store the processed splits
        processed_dataset = DatasetDict()

        # Process each split (train, validation, and test)
        for split_name in dataset.keys():
            print(f"Processing split: {split_name}")
            if split_name == 'test':
                dataset[split_name] = dataset[split_name].filter(lambda x: x['is_forecasting_user'] == True)

            # Pair sequences by user_id and add features for the current split
            paired_split = self._generate_collection_phase_sample_by_user_id(dataset[split_name], split_name, history_N)
            paired_split = self._create_text_column(paired_split)

            # Tokenization step
            def _preprocess(batch):
                    return tokenizer(
                        batch['text'], truncation=True, padding="max_length", max_length=max_length
                    )

            # Tokenize the paired dataset
            encoded = paired_split.map(_preprocess, batched=True)

            # Convert the 'state_change_valence or state_change_arousal' to 'labels'
            encoded = encoded.map(self._to_labels, batched=True, fn_kwargs={"from_col": f'disposition_change_{self.label}'})

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

        return {
            "mse": mse_val,
            "pearson": pearson_val,
            "r_squared": r2_val,
            "mae": mae_val,
            "rmse": rmse_val,
        }


class RobertaRegressionHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaRegressionHead, self).__init__()
        self.history_N = config.history_N
        self.extra_feature_size = config.history_N * 4 + 8
        self.dense = nn.Linear(config.hidden_size + self.extra_feature_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)


    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        B = x.size(0)
        x = self.dropout(x)

        valences = kwargs.get("valences").view(B, self.history_N)
        arousals = kwargs.get("arousals").view(B, self.history_N)

        valence_autocorrelation = kwargs.get("valence_autocorrelation").view(B, 1)
        valence_mssd = kwargs.get("valence_mssd").view(B, 1)
        valence_mean = kwargs.get("valence_mean").view(B, 1)

        arousal_autocorrelation = kwargs.get("arousal_autocorrelation").view(B, 1)
        arousal_mssd = kwargs.get("arousal_mssd").view(B, 1)
        arousal_mean = kwargs.get("arousal_mean").view(B, 1)

        valence_swings = kwargs.get("valence_swings").view(B, 1)
        bounce_to_inertia = kwargs.get("bounce_to_inertia").view(B, 1)

        valence_rolling_volatility = kwargs.get("valence_rolling_volatility").view(B, self.history_N)
        arousal_rolling_volatility = kwargs.get("arousal_rolling_volatility").view(B, self.history_N)

        # x = torch.zeros_like(x)  # mask out text features
        x = torch.cat([x,
                       valences,
                       arousals,
                       valence_mean,
                       valence_autocorrelation,
                       valence_mssd,
                       arousal_mean,
                       arousal_autocorrelation,
                       arousal_mssd,
                       valence_swings,
                       bounce_to_inertia,
                       valence_rolling_volatility,
                       arousal_rolling_volatility,
                       ], dim=-1)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForSequenceRegression(BertPreTrainedModel):
    # Use standard Roberta Config mapping
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, base_model_name: str, history_N: int):
        if isinstance(base_model_name, str):
            config = AutoConfig.from_pretrained(base_model_name)
        else:
            config = base_model_name

        super(RobertaForSequenceRegression, self).__init__(config)
        self.config = config
        self.config.num_labels = 1
        self.config.history_N = history_N

        # Load the base RobertaModel (with pretrained weights)
        self.roberta = AutoModel.from_pretrained(base_model_name, config=config, add_pooling_layer=False)
        # self.roberta.requires_grad=False  # freeze the body
        self.regressor = RobertaRegressionHead(config)

        # Initialize weights
        self.post_init()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None,
                valences=None, arousals=None,
                valence_mean=None, valence_autocorrelation=None, valence_mssd=None,
                arousal_mean=None, arousal_autocorrelation=None, arousal_mssd=None,
                valence_swings=None, bounce_to_inertia=None,
                valence_rolling_volatility=None, arousal_rolling_volatility=None,
                ):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        sequence_output = outputs[0]  # mask out all text features
        logits = self.regressor(features=sequence_output,
                                valences=valences, arousals=arousals,
                                valence_mean=valence_mean, valence_autocorrelation=valence_autocorrelation,
                                valence_mssd=valence_mssd,
                                arousal_mean=arousal_mean, arousal_autocorrelation=arousal_autocorrelation,
                                arousal_mssd=arousal_mssd,
                                valence_swings=valence_swings, bounce_to_inertia=bounce_to_inertia,
                                valence_rolling_volatility=valence_rolling_volatility,
                                arousal_rolling_volatility=arousal_rolling_volatility, )

        loss = None
        if labels is not None:
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # Extract features that need to be padded by the tokenizer
        # DataCollatorWithPadding expects a list of dicts with at least input_ids
        text_features = []
        for f in features:
            text_features.append({
                "input_ids": f["input_ids"],
                "attention_mask": f["attention_mask"]
            })

        # # Pad the text features
        batch = self.tokenizer.pad(
            text_features,
            padding=True,
            return_tensors="pt"
        )  # vectorization

        # Add the numeric features manually
        if "labels" in features[0]:
            batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.float32)
        
        # Add the extra features requested by user
        for key in ["valences", "arousals",
                    "valence_mean", "valence_autocorrelation", "valence_mssd",
                    "arousal_mean", "arousal_autocorrelation", "arousal_mssd",
                    "valence_swings", "bounce_to_inertia",
                    "valence_rolling_volatility", "arousal_rolling_volatility", ]:

            if key in features[0]:
                batch[key] = torch.tensor([f[key] for f in features], dtype=torch.float32)

        return batch


class RegressionTrainer:
    """
    Orchestrates tokenizer/model creation and fine-tuning for sequence regression with dual-sequence RoPE input.
    """

    def __init__(
            self,
            model_name: str,
            learning_rate: float,
            output_dir: str,
            train_bs: int,
            eval_bs: int,
            num_epochs: int,
            weight_decay: float,
            history_N: int,
            logging_dir: str = "./logs",
            report_to: str = "none",
            save_total_limit: int = 2,
            eval_strategy: str = "epoch",
            save_strategy: str = "epoch",
            run_name: str = "subtask2b",
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = RobertaForSequenceRegression(model_name, history_N)
        self.args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            learning_rate=learning_rate,
            per_device_train_batch_size=train_bs,
            per_device_eval_batch_size=eval_bs,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            save_total_limit=save_total_limit,
            logging_dir=logging_dir,
            report_to=report_to,
            run_name=run_name,
            logging_steps=50,
        )
        self._trainer = None

    def build_trainer(self, encoded: DatasetDict, metrics: MetricComputer) -> Trainer:
        """
        Create a Hugging Face Trainer for the regression task.
        """
        collator = CustomDataCollator(tokenizer=self.tokenizer)

        self._trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=encoded["train"],
            eval_dataset=encoded["validation"],
            compute_metrics=metrics.compute,
            data_collator=collator,
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
    parser.add_argument("--run_name", required=True, default="subtask2a", type=str, help="Name of the run.")
    parser.add_argument("--train", type=str, required=True, help="Path to the training dataset")
    parser.add_argument("--validation", type=str, help="Path to the validation dataset")
    parser.add_argument("--model", type=str, default="cardiffnlp/twitter-xlm-roberta-base", help="Pretrained model name")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for results")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--train_bs", type=int, default=16, help="Batch size for training")
    parser.add_argument("--eval_bs", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--report_to", type=str, default="none", help="Reporting options for logging")
    parser.add_argument("--max_length", type=int, default=512, help="Max length for tokenization")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the model")
    parser.add_argument("--label", type=str, default="state_change_valence", help="The target label name.")
    # parser.add_argument("--feature", type=str, default="valence", help="The valence or arousal feature.")
    parser.add_argument("--at_N", type=int, default=10,
                        help="The number of previous time steps to consider for history-based features.")
    args = parser.parse_args()

    data_files = {"train": args.train, "validation": args.validation}

    reg = RegressionTrainer(
        model_name=args.model,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        history_N=args.at_N,
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
        # feature=args.feature,
    )
    metrics = MetricComputer()

    # Prepare data
    encoded = pre.prepare(tokenizer=reg.tokenizer, max_length=args.max_length, history_N=args.at_N)

    # Build, train, evaluate
    trainer = reg.build_trainer(encoded, metrics)
    reg.fit()

    # eval_split_name = "test" if "test" in encoded else "validation"
    # results = reg.evaluate(encoded[eval_split_name])
    # print(f"{eval_split_name} results:", results)

    # Save final model if requested
    if args.save_dir:
        reg.save(args.save_dir)
        print(f"Final model and tokenizer saved in {args.save_dir}")


if __name__ == "__main__":
    main()