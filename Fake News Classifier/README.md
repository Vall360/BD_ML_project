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

Then run `trainer.py` to train the model. The script performs a small hyperparameter search using the Hugging Face `Trainer` and stores the best model together with logs and plots in `Results/`.

```bash
python trainer.py
```

Outputs generated in `Results/`:

- `model/` – directory with the trained model and tokenizer
- `log_history.csv` – log of metrics during training
- `test_metrics.csv` – metrics on the test split
- `*.png` – graphs of loss and metrics per epoch

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
