# Fake News Classifier

This folder contains utilities to train and run a BERT-based classifier for fake news detection. Training data is located in `Train Data/` and consists of CSV files with the following columns:

- `title` – news title
- `text` – full article text
- `label` – `1` for real news and `0` for fake

## Training

Install the required Python packages first:

```bash
pip install -r requirements.txt
```

If your Python build uses an SSL library older than OpenSSL 1.1.1 (for example
LibreSSL on macOS), you should keep `urllib3` below version 2.0. The provided
`requirements.txt` already pins `urllib3<2` to avoid SSL errors when downloading
models from Hugging Face.

Then run `trainer.py` to train the model. The script tokenizes the dataset in parallel using all available CPU cores. By default it performs a short hyperparameter search (two trials) with Optuna through the Hugging Face `Trainer`. The best model together with logs and plots will be stored in `Results/`. Pass `--no-optuna` to skip the hyperparameter search and train directly with the default settings.

```bash
python trainer.py
# or to skip the hyperparameter search
python trainer.py --no-optuna
```

Outputs generated in `Results/`:

- `model/` – directory with the trained model and tokenizer
- `log_history.csv` – log of metrics during training
- `test_metrics.csv` – metrics on the test split
- `*.png` – graphs of loss and metrics per epoch

On an Apple M2 with 16 GB RAM the full training (including hyperparameter search)
finishes in roughly four hours.

## Classical Baselines

To compare the BERT model with traditional approaches, run the baseline suite:

```bash
python baselines.py
# optionally tweak how much data is held out for evaluation
python baselines.py --eval-fraction 0.3
```

The script trains three models on the same splits:

- TF-IDF features with logistic regression
- Linear SVM on TF-IDF features
- A lightweight bidirectional LSTM

Metrics for each model are stored under `Results/baselines/<model>/metrics.json` so they
can be referenced in reports or papers. The script merges the provided training and
validation files, then stratifies a new validation split (default 30% of the combined data),
which typically yields validation F1 scores around 0.94–0.95 for the linear baselines.

## Inference

`inference.py` provides the `FakeNewsPredictor` class for fast prediction on a pandas `DataFrame`.

```python
from inference import FakeNewsPredictor
import pandas as pd

predictor = FakeNewsPredictor()
df = pd.DataFrame({"text": ["Example news text"]})
df["prediction"] = predictor.predict(df, "text")
```

The class loads the saved model from `Results/model` by default and processes the texts in batches for speed.
