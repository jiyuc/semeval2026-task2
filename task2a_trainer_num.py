import argparse
import numpy as np
from typing import Dict, Any
from datasets import load_dataset, DatasetDict, Dataset
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

from transformers import AutoConfig, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


# os.environ['WANDB_API_KEY'] = '1e99e9ffac3110635b72b4581c56dc7cf58c6fa4'

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="semeval2026-subtask2a"

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
    ):
        self.data_files = data_files
        self.label = label
        self.feature = feature


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


    def _generate_collection_phase_sample_by_user_id(self, dataset: Dataset) -> Dataset:
        """
        Pair consecutive sequences for each user and add features.
        Keep the last sequence but remove the `None` label (set it to `0`).
        """
        phase_samples = []
        df = pd.DataFrame(dataset)
        user_groups = df.groupby("user_id")  # group by each user

        for user_id, group in user_groups:
            collection_phases = group.groupby("collection_phase")  # group by each collection phase
            for phase_id, collection in collection_phases:
                collection = collection.sort_values(by=["timestamp"])  # sort by timestamp
                try:
                    assert len(collection) >= 3, f"Collection for user {user_id} in phase {phase_id} has less than 3 posts"
                except AssertionError:
                    # skip the current phase if it has less than 3 posts
                    continue

                # curr = collection.iloc[-1]  # the last post in the collection is treated as label
                prev = collection.iloc[-2]  # collect the second last post
                prev_prev = collection.iloc[-3] # collect the third last post

                phase_samples.append({
                    'user_id': user_id,
                    'collection_phase': phase_id,
                    # 'text1': prev_prev['text'].lower(),
                    'text2': prev['text'].lower(),
                    'state1': prev_prev[f'{self.feature}'],
                    'state2': prev[f'{self.feature}'],
                    'state_change': prev_prev[f'{self.label}'],
                    'time_diff': self._calculate_time_diff(prev_prev, prev),
                    f'{self.label}': prev[self.label]  # the state change in the last of each phase to predict
                })

        # Convert the processed user pairs back to a Hugging Face Dataset
        processed_df = pd.DataFrame(phase_samples)
        print(f"Number of generated phase samples: {len(processed_df)}")
        processed_dataset = Dataset.from_pandas(processed_df)
        return processed_dataset


    @staticmethod
    def combine_text(example):
        """
        Concatenate text1, state1, and text2 to create a new 'text' column.
        The columns are joined in the order: text1, state1, text2.
        """
        return {
            'text': f"state change: {example['state_change']} "
                    f"</s> time difference: {example['time_diff']} "
                    f"</s> previous state: {str(example['state1'])} "
                    f"</s> current state: {example['state2']} text: {str(example['text2'])}"
        }

    def _create_text_column(self, paired_dataset):
        """
        Apply the `combine_text` function to the dataset to create a new 'text' column.
        """
        return paired_dataset.map(self.combine_text, batched=False)

    def prepare(
            self,
            tokenizer,
            max_length: int,
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

            # Pair sequences by user_id and add features for the current split
            paired_split = self._generate_collection_phase_sample_by_user_id(dataset[split_name])
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
        self.dense = nn.Linear(config.hidden_size + 5, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        self.fnn = nn.Linear(4, 1)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        B = x.size(0)
        x = self.dropout(x)

        state1 = kwargs.get("state1", 0.0).view(B, 1)
        state2 = kwargs.get("state2", 0.0).view(B, 1)
        state_change = kwargs.get("state_change", 0.0).view(B, 1)
        time_diff = kwargs.get("time_diff", 0.0).view(B, 1)

        y = self.fnn(torch.cat([state1, state2, state_change, time_diff], dim=-1)) # use a layer to train relation


        x = torch.cat([x, time_diff, state1, state2, state_change, y], dim=-1)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForSequenceRegression(BertPreTrainedModel):
    # Use standard Roberta Config mapping
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, base_model_name: str):
        if isinstance(base_model_name, str):
            config = AutoConfig.from_pretrained(base_model_name)
        else:
            config = base_model_name

        super(RobertaForSequenceRegression, self).__init__(config)
        self.config = config
        self.config.num_labels = 1

        # Load the base RobertaModel (with pretrained weights)
        self.roberta = AutoModel.from_pretrained(base_model_name, config=config, add_pooling_layer=False)
        self.regressor = RobertaRegressionHead(config)

        # Initialize weights
        self.post_init()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None, state1=None, state2=None, state_change=None, time_diff=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        sequence_output = outputs[0]
        logits = self.regressor(sequence_output, state1=state1,
                                state2=state2,
                                state_change=state_change,
                                time_diff=time_diff)


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
            # padding=True,
            return_tensors="pt"
        )  # vectorization

        # Add the numeric features manually
        if "labels" in features[0]:
            batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.float32)
        
        # Add the extra features requested by user
        for key in ["state1", "state2", "state_change", "time_diff"]:
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
            logging_dir: str = "./logs",
            report_to: str = "none",
            metric_for_best: str = "mse",
            greater_is_better: bool = False,
            save_total_limit: int = 2,
            eval_strategy: str = "epoch",
            save_strategy: str = "epoch",
            run_name: str = "subtask2a",
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = RobertaForSequenceRegression(model_name)
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
            logging_steps=1,
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
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the model")
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

    # eval_split_name = "test" if "test" in encoded else "validation"
    # results = reg.evaluate(encoded[eval_split_name])
    # print(f"{eval_split_name} results:", results)

    # Save final model if requested
    # if args.save_dir:
    #     reg.save(args.save_dir)
    #     print(f"Final model and tokenizer saved in {args.save_dir}")


if __name__ == "__main__":
    main()