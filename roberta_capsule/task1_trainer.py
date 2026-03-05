import argparse
import numpy as np
from typing import Dict, Tuple, Any
from datasets import load_dataset, DatasetDict, concatenate_datasets, Features, Value
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    RobertaForSequenceClassification,
    PreTrainedModel,
    AutoConfig,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import evaluate
import os
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import torch

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="semeval2026-subtask1"

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="end"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"


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
        label: str,
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

    def load(self) -> DatasetDict:
        """
        Load dataset from CSV files into a DatasetDict.

        Returns:
            A DatasetDict with 'train'/'validation'/optional 'test' splits.
        """
        dataset = load_dataset("csv", data_files=self.data_files)


        dataset = dataset.map(lambda e: {"text": e["text"].lower()})
        return dataset


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
        max_length: int,
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



# Modify the model to extract the [CLS] token and add a regression head
class RoBERTaWithRegressionHead(PreTrainedModel):
    """
    Apply a joint modeling approach to RoBERTa for regression with temporal scaling.
    """
    def __init__(self, base_model_name: str, dropout_rate: float):
        super().__init__(AutoConfig.from_pretrained(base_model_name))
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.backbone = AutoModelForSequenceClassification.from_pretrained(base_model_name,
                                                                    num_labels=self.config.num_labels,
                                                                )
        self.config.hidden_dropout_prob = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.regression_head = nn.Linear(self.backbone.config.hidden_size, 1)  # Regression head

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # Pass inputs through roberta
        outputs = self.backbone.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)

        # Get the last hidden state (of shape [batch_size, seq_length, hidden_size])
        last_hidden_state = outputs.last_hidden_state

        # Extract the hidden state corresponding to the [CLS] token (index 0)
        cls_hidden_state = last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size]

        # Apply dropout
        cls_hidden_state = self.dropout(cls_hidden_state)

        # Pass the [CLS] token hidden state through the regression head
        logits = self.regression_head(cls_hidden_state).mean(dim=1).squeeze(-1)

        loss = None
        if labels is not None:
            # Compute MSE loss
            loss = nn.functional.mse_loss(logits.view(-1), labels.view(-1).float())

        # If labels are provided, calculate and return the loss
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )



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

        Args:
            eval_pred: Tuple(preds, labels) as provided by HF Trainer.

        Returns:
            Dict of scalar metrics.
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


class RegressionTrainer:
    """
    Orchestrates tokenizer/model creation and fine-tuning for sequence regression.
    """

    def __init__(
        self,
        model_name: str,
        dropout_rate: float,
        output_dir: str = "./results",
        learning_rate: float = 2e-5,
        train_bs: int = 16,
        num_epochs: int = 4,
        weight_decay: float = 0.01,
        logging_dir: str = "./logs",
        report_to: str = "none",
        save_total_limit: int = 2,
        eval_strategy: str = "no",
        save_strategy: str = "epoch",
        run_name: str = "subtask1",
    ):
        """
        Configure model, tokenizer, and training arguments.

        Args:
            model_name: Pretrained model identifier or local path.
            output_dir: Directory for checkpoints and outputs.
            learning_rate: Optimizer learning rate.
            train_bs: Per-device train batch size..
            num_epochs: Number of training epochs.
            weight_decay: Weight decay.
            logging_dir: Directory for logs.
            report_to: Reporting integrations ("none", "tensorboard", "wandb", etc.).
            save_total_limit: Max number of checkpoints to keep.
            eval_strategy: Evaluation frequency ("no", "steps", "epoch").
            save_strategy: Saving frequency ("no", "steps", "epoch").
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"continual sft on {model_name} ...")
        self.model = RoBERTaWithRegressionHead(self.model_name,
                                               dropout_rate=dropout_rate)

        self.args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            learning_rate=learning_rate,
            per_device_train_batch_size=train_bs,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            save_total_limit=save_total_limit,
            logging_dir=logging_dir,
            report_to=report_to,
            logging_steps=50,
            run_name=run_name,
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
        )
        return self._trainer

    def fit(self):
        """
        Run fine-tuning using the built Trainer.
        """
        if self._trainer is None:
            raise RuntimeError("Trainer is not built. Call build_trainer() first.")
        self._trainer.train()


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
            --validation data/SemEval2026/data/split/pred_subtask1.csv \
            --model cardiffnlp/twitter-xlm-roberta-base \
            --output_dir ./results \
            --save_dir ./xlm_valence_reg_model
    """
    parser = argparse.ArgumentParser(description="Fine-tune a regression model on valence/arousal data.")
    parser.add_argument("--run_name", required=True, default="subtask1", type=str, help="Name of the run.")
    parser.add_argument("--train", required=True, help="Path to training CSV.")
    parser.add_argument("--model", default="cardiffnlp/twitter-xlm-roberta-base", help="HF model name or local path.")
    parser.add_argument("--output_dir", default="./results", help="Trainer output directory.")
    parser.add_argument("--save_dir", default="./xlm_valence_reg_model", help="Directory to save final model.")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--train_bs", type=int, default=16, help="Per-device train batch size.")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--report_to", default="none", help="Reporting integration for Trainer.")
    parser.add_argument("--label", type=str, default="valence", help="The target label name.")
    args = parser.parse_args()

    data_files = {"train": args.train}

    # Build components
    reg = RegressionTrainer(
        model_name=args.model,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        train_bs=args.train_bs,
        num_epochs=args.epochs,
        weight_decay=args.weight_decay,
        report_to=args.report_to,
        run_name=args.run_name,
        dropout_rate=args.dropout_rate
    )
    pre = DataPreprocessor(
        data_files=data_files,
        label=args.label,
    )

    # Prepare data
    encoded = pre.prepare(tokenizer=reg.tokenizer, max_length=args.max_length)

    # define metrics
    metrics = MetricComputer()

    # Build, train, evaluate
    _ = reg.build_trainer(encoded, metrics)
    reg.fit()

    # Save artifacts
    reg.save(args.save_dir)
    print(f"Final model and tokenizer saved in {args.save_dir}")

if __name__ == "__main__":
    main()