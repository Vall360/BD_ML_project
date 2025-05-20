# Fake News BERT Pipeline

This project provides a small set of modules to collect news
articles, train a BERT-based fake news classifier and deploy the
resulting model.

## Modules

- `fake_news.data_parser.DataParser` – fetch or load news articles into a
  `pandas.DataFrame`. Articles can be retrieved from the NewsAPI service
  using a query of keywords or loaded from a prepared dataset.
- `fake_news.trainer.FakeNewsTrainer` – fine-tunes a BERT model using the
  `transformers` library.
- `fake_news.deployment.FakeNewsClassifier` – loads a trained model and
  returns predictions for new text inputs.

## Usage

```python
from fake_news import DataParser, FakeNewsTrainer, FakeNewsClassifier

# 1) Collect data
parser = DataParser(api_key="YOUR_NEWSAPI_KEY")
train_df = parser.fetch_news(["politics", "economy"], limit=200)
# Add labels to `train_df` as needed

# 2) Train model
trainer = FakeNewsTrainer(model_name="distilbert-base-uncased")
trainer.train(train_df)

# 3) Deploy and classify
clf = FakeNewsClassifier(model_dir="model")
results = clf.classify(["Some headline", "Another article text"])
```

The environment for training requires the packages `pandas`,
`requests`, `datasets` and `transformers`.
