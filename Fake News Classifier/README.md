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

Run `trainer.py` to train the model. By default the script trains once using preset hyperparameters. Pass `--search` to run an Optuna hyperparameter search (two trials by default). Tokenization uses multiple CPU cores and all results are stored in `Results/`.

```bash
python trainer.py
```

To perform a search with two trials:

```bash
python trainer.py --search --trials 2
```

Outputs generated in `Results/`:

- `model/` – directory with the trained model and tokenizer
- `log_history.csv` – log of metrics during training
- `test_metrics.csv` – metrics on the test split
- `*.png` – graphs of loss and metrics per epoch

On an Apple M2 with 16 GB RAM the base training run finishes in roughly four hours. Running a hyperparameter search will take proportionally longer depending on the number of trials.

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
