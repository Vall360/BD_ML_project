import os
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import argparse
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'Train Data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'Results')
MODEL_DIR = os.path.join(RESULTS_DIR, 'model')

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_split(filename):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, delimiter=';')
    df['text'] = (df['title'].fillna('') + '. ' + df['text'].fillna('')).str.strip()
    return Dataset.from_pandas(df[['text', 'label']])


def load_datasets():
    train_ds = load_split('train (2).csv')
    eval_ds = load_split('evaluation.csv')
    test_ds = load_split('test (1).csv')
    return train_ds, eval_ds, test_ds


TOKENIZER_NAME = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# Limit the number of processes used during preprocessing to avoid excessive
# memory usage on systems with limited RAM (e.g. 16GB on Mac M2).
NUM_PROC = max(1, min(4, multiprocessing.cpu_count()))


def tokenize(batch):
    return tokenizer(batch['text'], truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def model_init():
    return AutoModelForSequenceClassification.from_pretrained(TOKENIZER_NAME, num_labels=2)



def main():
    parser = argparse.ArgumentParser(description="Train fake news classifier")
    parser.add_argument(
        "--search",
        action="store_true",
        help="run Optuna hyperparameter search before training",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=2,
        help="number of Optuna trials when --search is enabled",
    )
    args = parser.parse_args()

    train_ds, eval_ds, test_ds = load_datasets()
    train_ds = train_ds.map(tokenize, batched=True, num_proc=NUM_PROC)
    eval_ds = eval_ds.map(tokenize, batched=True, num_proc=NUM_PROC)
    test_ds = test_ds.map(tokenize, batched=True, num_proc=NUM_PROC)

    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        dataloader_num_workers=NUM_PROC,
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if args.search:
        def hp_space(trial):
            return {
                'learning_rate': trial.suggest_float('learning_rate', 2e-5, 5e-5, log=True),
                'num_train_epochs': trial.suggest_int('num_train_epochs', 2, 4),
                'per_device_train_batch_size': trial.suggest_categorical('per_device_train_batch_size', [16, 32]),
            }

        best_run = trainer.hyperparameter_search(
            direction='maximize',
            hp_space=hp_space,
            n_trials=args.trials,
            compute_objective=lambda metrics: metrics.get('eval_f1', 0.0),
        )
        for n, v in best_run.hyperparameters.items():
            setattr(trainer.args, n, v)

    trainer.train()
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    # Save logs
    log_history = pd.DataFrame(trainer.state.log_history)
    log_history.to_csv(os.path.join(RESULTS_DIR, 'log_history.csv'), index=False)

    # Plot metrics
    def plot_metric(metric):
        df = log_history.dropna(subset=['epoch', metric])
        if df.empty:
            return
        plt.figure()
        plt.plot(df['epoch'], df[metric])
        plt.xlabel('epoch')
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'{metric}.png'))
        plt.close()

    for m in ['loss', 'eval_loss', 'eval_accuracy', 'eval_f1']:
        plot_metric(m)

    # Evaluate on test set
    test_metrics = trainer.evaluate(test_ds)
    pd.DataFrame([test_metrics]).to_csv(os.path.join(RESULTS_DIR, 'test_metrics.csv'), index=False)


if __name__ == '__main__':
    main()
