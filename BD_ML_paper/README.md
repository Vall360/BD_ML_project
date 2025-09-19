# BD_ML_paper

This folder contains a standalone reimplementation of the original Colab notebook used in the BD&ML project. The code was ported to plain Python scripts so the full pipeline can be executed locally without modifying the notebook logic.

## Contents
- `ml_pipeline.py` – core module that mirrors the notebook (data collection helpers, `ML_build` class, preprocessing utilities, beta calculation, etc.).
- `run_regressions.py` – convenience script that drives the full workflow and estimates the research regressions.
- `News_dataset.csv` – raw Google News export (no FinBERT/FakeNews scores).
- `BERT_project_results.csv` – cache of FinBERT/FakeNews/NER outputs (created automatically after the first run).

## Pipeline summary
1. **News ingestion** – `ml_pipeline.load_news_dataset` reproduces the notebook loader and normalises the raw CSV (dates parsed, column names aligned).
2. **Initial clean-up** – `ML_build.data_preparation` drops duplicates, removes non-ASCII titles, and splits the publisher out of the title string if needed.
3. **BERT feature extraction** – `ML_build.BERT_sentiment_test`, `ML_build.BERT_quality_test`, and `ML_build.BERT_NER_test` call the Hugging Face pipelines used in the notebook (FinBERT, fake-news classifier, and NER). Results are persisted to `BERT_project_results.csv` so subsequent runs can skip the expensive inference step.
4. **Encoding** – `ML_build.One_hot_encode` recreates the one-hot / manual encodings for FinBERT and FakeNews outcomes and the interaction terms (`positive*Label_1`, `negative*Label_1`).
5. **Market data** – `ML_build.import_stock_data` pulls OHLCV, dividends, S&P500 returns, earnings surprise, and market cap proxies through `yfinance`, matching the four-year window used in the notebook.
6. **Beta & abnormal returns** – `ML_build.Beta_calculation` fits per-company CAPM regressions (returns vs S&P500) via `statsmodels`, storing betas, alphas, and abnormal returns (`ar`).
7. **Panel aggregation** – `ML_build.prepare_news_df` aggregates daily news features by company, derives the counts/ratios required by the regressions (e.g. `positive_label_1`, `rel_neg_label_0`), joins them with the market data, and ensures panel structure with a `const` column.
8. **(Optional)** `ML_build.prepare_final`, `train_test_split`, `LR_class`, and `RF_class` remain available exactly as in the notebook for classification experiments.

## Running the full workflow
1. Install the dependencies used in the notebook (Python ≥3.9 recommended):
   ```bash
   pip install pandas numpy tqdm yfinance datasets transformers torch optuna linearmodels stargazer statsmodels scikit-learn
   ```
   Additional packages (`feedparser`, `dateparser`, `bs4`, `spacy`) are only needed for news scraping / DBSCAN replication.
2. Ensure you are inside the project directory and run the driver script:
   ```bash
   cd BD_ML_paper
   python run_regressions.py
   ```
   - On the first execution the script will run the FinBERT/FakeNews/NER models and write `BERT_project_results.csv` (this can take a long time and downloads the models the first time).
   - Subsequent executions reuse the cached CSV and skip model inference.
3. The script builds the panel dataframe, estimates the eight `PanelOLS` specifications listed in the research workflow, prints the regression tables to stdout, and writes a Stargazer HTML report to `Report0.html`.

## Behaviour notes
- All parameters are hard-coded to stay aligned with the notebook (no `argparse`).
- File paths are constructed relative to `ml_pipeline.py`, so the scripts can be executed from any working directory as long as the project layout is preserved.
- Network access is required for Hugging Face model downloads and `yfinance` data pulls. If cached data already exists, the run regains determinism.
- `ml_pipeline.py` also exposes the original helper routines (`GoogleNews`, `get_news`, DBSCAN story flagging, Optuna tuning, etc.) so exploratory workflows from the notebook remain available.

## Output
- `Report0.html` – Stargazer summary of the eight panel regressions.
- Intermediate dataframes (`ML_build.df_news`, `ML_build.df_stock`, `ML_build.model_run_final`) are accessible through the class instance for debugging or further analysis.

## Troubleshooting
- If `yfinance` or Hugging Face calls fail because of rate limits or missing network permissions, rerun the script once connectivity is restored.
- To refresh the BERT features, delete `BERT_project_results.csv` (or move it aside) and rerun `run_regressions.py`; the script will recompute and recreate the cache automatically.
