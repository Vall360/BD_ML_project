"""BERT model trainer for fake news classification."""

from __future__ import annotations

import pandas as pd
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)


class FakeNewsTrainer:
    """Fine-tunes a BERT-like model for fake news detection."""

    def __init__(self, model_name: str = "distilbert-base-uncased",
                 output_dir: str = "model", num_labels: int = 2) -> None:
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def _tokenize(self, examples):
        return self.tokenizer(examples["text"], truncation=True)

    def train(self, df_train: pd.DataFrame, df_val: pd.DataFrame | None = None,
              *, epochs: int = 3, batch_size: int = 8, lr: float = 2e-5) -> None:
        """Train the model using the provided data frames."""
        train_ds = Dataset.from_pandas(df_train)
        train_ds = train_ds.map(self._tokenize, batched=True)

        eval_ds = None
        if df_val is not None:
            eval_ds = Dataset.from_pandas(df_val)
            eval_ds = eval_ds.map(self._tokenize, batched=True)

        data_collator = DataCollatorWithPadding(self.tokenizer)
        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            evaluation_strategy="epoch" if eval_ds is not None else "no",
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        trainer.train()
        trainer.save_model(self.output_dir)
