import os
import sys
from pathlib import Path
from packaging import version

# 0. Confirm we're using the stock Colab environment
MIN_VERSIONS = {
  "torch": "2.5.0",
  "transformers": "4.46.0",
  "datasets": "3.0.0",
  "accelerate": "1.0.1",
  "pandas": "2.2.2",
  "scikit_learn": "1.5.0",
  "matplotlib": "3.8.0",
  "seaborn": "0.13.0",
}
missing = []
for pkg, min_ver in MIN_VERSIONS.items():
  try:
      mod = __import__(pkg if pkg != "scikit_learn" else "sklearn")
      actual = mod.__version__
      if version.parse(actual) < version.parse(min_ver):
          missing.append(f"{pkg}>={min_ver} (found {actual})")
  except ImportError:
      missing.append(f"{pkg}>={min_ver} (not installed)")
if missing:
  raise ImportError(
      "The following dependencies are too old or missing in this runtime:\n"
      + "\n".join(missing)
      + "\nUse a fresh Colab GPU runtime or pip install them manually."
  )

import gc
import json
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
  AutoModelForSequenceClassification,
  AutoTokenizer,
  DataCollatorWithPadding,
  Trainer,
  TrainingArguments,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sns.set_theme()

DATA_DIR = Path("/content/news_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
for csv_name in ("Fake.csv", "True.csv"):
  src = Path("/content") / csv_name
  if not src.exists():
      raise FileNotFoundError(
          f"Upload {csv_name} into /content (use the Colab file uploader) before running this cell."
      )
  dst = DATA_DIR / csv_name
  dst.write_bytes(src.read_bytes())
print("Data files:", list(DATA_DIR.glob("*.csv")))

def load_dataset(data_dir: Path) -> pd.DataFrame:
  fake = pd.read_csv(data_dir / "Fake.csv")
  true = pd.read_csv(data_dir / "True.csv")
  fake["label"] = 0
  true["label"] = 1
  df = pd.concat([fake, true], ignore_index=True)
  for col in ("subject", "title", "text", "date"):
      df[col] = df[col].fillna("").astype(str).str.strip()
  df["publish_datetime"] = pd.to_datetime(df["date"], errors="coerce")
  df["publish_year"] = df["publish_datetime"].dt.year
  title_or_subject = df["title"].mask(df["title"] == "", df["subject"])
  combined = (title_or_subject + ". " + df["text"]).str.strip()
  df["text_combined"] = combined.mask(combined.str.startswith(". "), df["text"])
  return df

def build_hf_dataset(df: pd.DataFrame) -> Dataset:
  return Dataset.from_pandas(
      df[["text_combined", "label"]].rename(columns={"text_combined": "text"}),
      preserve_index=False,
  )

def compute_metrics(eval_pred):
  logits, labels = eval_pred
  preds = np.argmax(logits, axis=1)
  precision, recall, f1, _ = precision_recall_fscore_support(
      labels, preds, average="binary", zero_division=0
  )
  accuracy = accuracy_score(labels, preds)
  return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def stratified_split(df: pd.DataFrame, val_fraction: float, seed: int):
  stratify = df["label"] if df["label"].nunique() > 1 else None
  train_df, eval_df = train_test_split(
      df, test_size=val_fraction, stratify=stratify, random_state=seed
  )
  return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)

def train_fold(train_df, eval_df, test_df, cfg, fold_dir: Path):
  fold_dir.mkdir(parents=True, exist_ok=True)
  tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
  data_collator = DataCollatorWithPadding(tokenizer)

  def tokenize(batch):
      return tokenizer(batch["text"], truncation=True, max_length=cfg["max_length"])

  train_ds = build_hf_dataset(train_df).map(tokenize, batched=True, num_proc=cfg["num_proc"])
  eval_ds = build_hf_dataset(eval_df).map(tokenize, batched=True, num_proc=cfg["num_proc"])
  test_ds = build_hf_dataset(test_df).map(tokenize, batched=True, num_proc=cfg["num_proc"])

  def model_init():
      return AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=2)

  training_args = TrainingArguments(
      output_dir=str(fold_dir / "hf_runs"),
      evaluation_strategy="epoch",
      logging_strategy="epoch",
      save_strategy="no",
      num_train_epochs=cfg["epochs"],
      per_device_train_batch_size=cfg["batch_size"],
      per_device_eval_batch_size=cfg["batch_size"],
      gradient_accumulation_steps=cfg["grad_accum"],
      learning_rate=cfg["learning_rate"],
      max_grad_norm=cfg["max_grad_norm"],
      warmup_ratio=cfg["warmup_ratio"],
      report_to=[],
      dataloader_num_workers=max(1, cfg["num_proc"]),
      load_best_model_at_end=False,
      seed=cfg["seed"],
  )

  trainer = Trainer(
      model_init=model_init,
      args=training_args,
      train_dataset=train_ds,
      eval_dataset=eval_ds,
      tokenizer=tokenizer,
      data_collator=data_collator,
      compute_metrics=compute_metrics,
  )

  trainer.train()
  eval_metrics = trainer.evaluate(eval_ds)
  test_metrics = trainer.evaluate(test_ds)

  pd.DataFrame(trainer.state.log_history).to_csv(fold_dir / "log_history.csv", index=False)
  with (fold_dir / "eval_metrics.json").open("w") as f:
      json.dump(eval_metrics, f, indent=2)
  with (fold_dir / "test_metrics.json").open("w") as f:
      json.dump(test_metrics, f, indent=2)

  metrics = {
      "train_samples": len(train_df),
      "eval_samples": len(eval_df),
      "test_samples": len(test_df),
      "eval_accuracy": float(eval_metrics.get("eval_accuracy", 0.0)),
      "eval_f1": float(eval_metrics.get("eval_f1", 0.0)),
      "test_accuracy": float(test_metrics.get("eval_accuracy", 0.0)),
      "test_f1": float(test_metrics.get("eval_f1", 0.0)),
  }
  del trainer, train_ds, eval_ds, test_ds
  gc.collect()
  if torch.cuda.is_available():
      torch.cuda.empty_cache()
  return metrics

def run_time_validation(df, cfg, results_dir):
  rows = []
  years = sorted(df["publish_year"].dropna().unique())
  for year in years:
      if cfg["time_start_year"] and year < cfg["time_start_year"]:
          continue
      if cfg["time_end_year"] and year > cfg["time_end_year"]:
          continue
      train_df = df[df["publish_year"] < year]
      test_df = df[df["publish_year"] == year]
      if len(train_df) < cfg["min_train_size"] or len(test_df) < cfg["min_test_size"]:
          continue
      train_split, eval_split = stratified_split(train_df, cfg["val_fraction"], cfg["seed"])
      fold_dir = results_dir / "time_splits" / f"time_train_le_{year-1}_test_{year}"
      metrics = train_fold(train_split, eval_split, test_df, cfg, fold_dir)
      rows.append({
          "fold": fold_dir.name,
          "description": f"Train <= {year-1}, test {year}",
          "test_year": year,
          **metrics,
      })
      if cfg["time_max_folds"] and len(rows) >= cfg["time_max_folds"]:
          break
  if not rows:
      print("No time folds met size thresholds.")
      return
  df_rows = pd.DataFrame(rows).sort_values("test_year")
  df_rows.to_csv(results_dir / "time_metrics.csv", index=False)
  plt.figure(figsize=(8, 4))
  plt.plot(df_rows["test_year"], df_rows["test_f1"], marker="o", label="F1")
  plt.plot(df_rows["test_year"], df_rows["test_accuracy"], marker="s", label="Accuracy")
  plt.xlabel("Test year")
  plt.ylabel("Score")
  plt.legend()
  plt.tight_layout()
  plt.savefig(results_dir / "time_performance.png", dpi=300)
  plt.close()
  print("Saved time metrics.")

def select_subject_domains(df, cfg):
  true_subjects = (
      df.loc[df["label"] == 1, "subject"]
      .value_counts()
      .loc[lambda s: s >= cfg["subject_min_samples"]]
      .head(cfg["subject_top_true"])
      .index.tolist()
  )
  fake_subjects = (
      df.loc[df["label"] == 0, "subject"]
      .value_counts()
      .loc[lambda s: s >= cfg["subject_min_samples"]]
      .head(cfg["subject_top_fake"])
      .index.tolist()
  )
  domains = []
  for true_s in true_subjects:
      for fake_s in fake_subjects:
          idx = df.index[(df["subject"] == true_s) | (df["subject"] == fake_s)]
          if len(idx) < cfg["subject_min_samples"] * 2:
              continue
          domains.append((true_s, fake_s, idx))
  return domains

def run_subject_validation(df, cfg, results_dir):
  domains = select_subject_domains(df, cfg)
  if not domains:
      print("No subject domains satisfy the sample threshold.")
      return
  rows = []
  for i, (train_true, train_fake, train_idx) in enumerate(domains):
      for j, (test_true, test_fake, test_idx) in enumerate(domains):
          if i == j:
              continue
          train_df = df.loc[train_idx]
          test_df = df.loc[test_idx]
          train_split, eval_split = stratified_split(train_df, cfg["val_fraction"], cfg["seed"])
          fold_dir = results_dir / "subject_splits" / f"{train_true}_vs_{train_fake}__to__{test_true}_vs_{test_fake}"
          metrics = train_fold(train_split, eval_split, test_df, cfg, fold_dir)
          rows.append({
              "fold": fold_dir.name,
              "train_domain": f"{train_true} vs {train_fake}",
              "test_domain": f"{test_true} vs {test_fake}",
              **metrics,
          })
          if cfg["subject_max_pairs"] and len(rows) >= cfg["subject_max_pairs"]:
              break
      if cfg["subject_max_pairs"] and len(rows) >= cfg["subject_max_pairs"]:
          break
  if not rows:
      print("Subject study skipped (max_pairs reached or insufficient domains).")
      return
  df_rows = pd.DataFrame(rows)
  df_rows.to_csv(results_dir / "subject_metrics.csv", index=False)
  pivot = df_rows.pivot(index="train_domain", columns="test_domain", values="test_f1")
  plt.figure(figsize=(min(10, 2 + pivot.shape[1]), min(8, 2 + pivot.shape[0])))
  sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
  plt.title("Subject-domain F1")
  plt.tight_layout()
  plt.savefig(results_dir / "subject_heatmap.png", dpi=300)
  plt.close()
  print("Saved subject metrics.")

CONFIG = {
  "model_name": "distilbert-base-uncased",
  "epochs": 2,
  "batch_size": 8,
  "grad_accum": 2,
  "learning_rate": 2e-5,
  "max_grad_norm": 1.0,
  "warmup_ratio": 0.05,
  "val_fraction": 0.1,
  "max_length": 256,
  "num_proc": 1,
  "seed": 42,
  "mode": "both",  # "time", "subject", or "both"
  "time_start_year": 2016,
  "time_end_year": 2018,
  "min_train_size": 3000,
  "min_test_size": 500,
  "time_max_folds": 2,          # increase for more years once stable
  "subject_top_true": 2,
  "subject_top_fake": 2,
  "subject_min_samples": 800,
  "subject_max_pairs": 4,
}

RESULTS_DIR = Path("/content/domain_reports")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df = load_dataset(DATA_DIR)
if CONFIG["mode"] in {"time", "both"}:
  run_time_validation(df, CONFIG, RESULTS_DIR)
if CONFIG["mode"] in {"subject", "both"}:
  run_subject_validation(df, CONFIG, RESULTS_DIR)
print("Artifacts stored in", RESULTS_DIR)