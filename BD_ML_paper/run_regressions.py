"""Execute the BD&ML pipeline end-to-end and run panel regressions.

This script relies on :mod:`ml_pipeline` to reproduce the original notebook
logic and then estimates the panel models specified in the research workflow.
All parameters are hard-coded to match the requested execution order.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from linearmodels.panel import PanelOLS
from stargazer.stargazer import Stargazer

from ml_pipeline import (
    BERT_RESULTS_PATH,
    NEWS_DATA_PATH,
    ML_build,
    load_news_dataset,
)


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the second level of the MultiIndex is a ``DatetimeIndex``."""

    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("DataFrame must have a MultiIndex [symbol, Date].")
    tuples = [
        (entity, pd.to_datetime(date))
        for entity, date in df.index.to_list()
    ]
    df = df.copy()
    df.index = pd.MultiIndex.from_tuples(tuples, names=["symbol", "Date"])
    return df


def run_pipeline() -> pd.DataFrame:
    """Run the end-to-end pipeline and return the panel-ready dataframe."""

    news_df = load_news_dataset(NEWS_DATA_PATH)

    ml = ML_build()
    ml.news_df(news_df)
    ml.load_BERT(BERT_RESULTS_PATH)
    '''if BERT_RESULTS_PATH.exists():
        ml.load_BERT(BERT_RESULTS_PATH)
    else:
        ml.data_preparation()
        ml.BERT_sentiment_test()
        ml.BERT_quality_test()
        ml.BERT_NER_test()
        ml.save_BERT(BERT_RESULTS_PATH)'''

    ## Raises an error + we already do it before BERT model results
    ##ml.data_preparation()

    ml.One_hot_encode(Interactions=True, OHE=False)
    ml.import_stock_data()
    ml.Beta_calculation()

    model_run_final = ml.prepare_news_df()
    panel_df = ensure_datetime_index(model_run_final)
    return panel_df


def run_regressions(panel_df: pd.DataFrame) -> None:
    """Estimate the specified set of panel regressions and export Stargazer."""

    formulas = [
        "ar ~ const + dividends + volume + market_cap + surprise_percent + positive_label_1 + negative_label_1 + EntityEffects",
        "ar ~ const + dividends + volume + market_cap + surprise_percent + positive_label_1 + negative_label_1 + positive_label_0 + EntityEffects",
        "ar ~ const + dividends + volume + market_cap + surprise_percent + positive_label_1 + negative_label_1 + negative_label_0 + EntityEffects",
        "ar ~ const + dividends + volume + market_cap + surprise_percent + positive_label_1 + negative_label_1 + positive_label_0 + negative_label_0 + EntityEffects",
        "ar ~ const + dividends + volume + market_cap + surprise_percent + rel_pos_label_1 + rel_neg_label_1 + EntityEffects",
        "ar ~ const + dividends + volume + market_cap + surprise_percent + rel_pos_label_1 + rel_neg_label_1 + rel_pos_label_0 + EntityEffects",
        "ar ~ const + dividends + volume + market_cap + surprise_percent + rel_pos_label_1 + rel_neg_label_1 + rel_neg_label_0 + EntityEffects",
        "ar ~ const + dividends + volume + market_cap + surprise_percent + rel_pos_label_1 + rel_neg_label_1 + rel_pos_label_0 + rel_neg_label_0 + EntityEffects",
    ]

    results = []
    for formula in formulas:
        model = PanelOLS.from_formula(
            formula,
            data=panel_df,
            check_rank=False,
            drop_absorbed=True,
        )
        fitted = model.fit(cov_type="clustered", cluster_entity=True)
        print(fitted)
        results.append(fitted)

    report = Stargazer(results)
    html = report.render_html()
    output_path = Path("Report0.html")
    output_path.write_text(html, encoding="utf-8")
    print(f"Saved Stargazer report to {output_path.resolve()}")


if __name__ == "__main__":
    df = run_pipeline()
    run_regressions(df)
