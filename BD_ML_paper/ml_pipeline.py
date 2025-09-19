"""Reimplementation of the BD&ML project Colab notebook as a Python module.

This module mirrors the logic from the original notebook, adapting it so the
pipeline can be executed locally. The code keeps the original method names and
structure (e.g. the ``ML_build`` class) while replacing Colab-specific
operations (``!pip`` installs, Google Drive copies) with standard Python
alternatives. No behavioural changes are introduced beyond those required to
run outside of Colab.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import yfinance as yf

# The original notebook calls ``tqdm.pandas()`` globally to enable progress bars
# for ``progress_apply``. We keep the same behaviour here.
tqdm.pandas()

# Local paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
NEWS_DATA_PATH = BASE_DIR / "News_dataset.csv"
BERT_RESULTS_PATH = BASE_DIR / "BERT_project_results.csv"
SYMBOL_LIST_PATH = BASE_DIR / "symbol_list_selected.csv"


# Google News helper ----------------------------------------------------------
class GoogleNews:
    """Port of the Google News helper class defined in the notebook."""

    def __init__(self, lang: str = "en", country: str = "USA") -> None:
        self.lang = lang.lower()
        self.country = country.upper()
        self.BASE_URL = "https://news.google.com/rss"

    def __top_news_parser(self, text: str) -> List[Dict[str, str]]:
        from bs4 import BeautifulSoup  # Imported lazily to match notebook usage

        try:
            bs4_html = BeautifulSoup(text, "html.parser")
            lis = bs4_html.find_all("li")
            sub_articles: List[Dict[str, str]] = []
            for li in lis:
                try:
                    sub_articles.append({
                        "url": li.a["href"],
                        "title": li.a.text,
                        "publisher": li.font.text,
                    })
                except Exception:
                    pass
            return sub_articles
        except Exception:
            return text  # type: ignore[return-value]

    def __ceid(self) -> str:
        return f"?ceid={self.country}:{self.lang}&hl={self.lang}&gl={self.country}"

    def __add_sub_articles(self, entries: Iterable[dict]) -> List[dict]:
        entries = list(entries)
        for i, _ in enumerate(entries):
            if "summary" in entries[i].keys():
                entries[i]["sub_articles"] = self.__top_news_parser(entries[i]["summary"]).copy()
            else:
                entries[i]["sub_articles"] = None
        return entries

    def __scaping_bee_request(self, api_key: str, url: str):
        import requests

        response = requests.get(
            url="https://app.scrapingbee.com/api/v1/",
            params={
                "api_key": api_key,
                "url": url,
                "render_js": "false",
            },
        )
        if response.status_code == 200:
            return response
        msg = f"ScrapingBee status_code: {response.status_code} {response.text}"
        raise RuntimeError(msg)

    def __parse_feed(self, feed_url: str, proxies: Optional[dict] = None, scraping_bee: Optional[str] = None) -> Dict[str, list]:
        import feedparser
        import requests

        if scraping_bee and proxies:
            raise RuntimeError("Pick either ScrapingBee or proxies. Not both!")

        if scraping_bee:
            response = self.__scaping_bee_request(api_key=scraping_bee, url=feed_url)
        else:
            response = requests.get(feed_url, proxies=proxies)

        if "https://news.google.com/rss/unsupported" in response.url:
            raise RuntimeError("This feed is not available")

        parsed = feedparser.parse(response.text)

        if not scraping_bee and not proxies and len(parsed["entries"]) == 0:
            parsed = feedparser.parse(feed_url)

        return {k: parsed[k] for k in ("feed", "entries")}

    def __search_helper(self, query: str) -> str:
        import urllib.parse

        return urllib.parse.quote_plus(query)

    def __from_to_helper(self, validate: Optional[str] = None) -> str:
        from dateparser import parse as parse_date

        try:
            parsed = parse_date(validate).strftime("%Y-%m-%d")
            return str(parsed)
        except Exception as exc:  # pragma: no cover - mimic original behaviour
            raise RuntimeError("Could not parse your date") from exc

    def top_news(self, proxies: Optional[dict] = None, scraping_bee: Optional[str] = None) -> Dict[str, list]:
        data = self.__parse_feed(self.BASE_URL + self.__ceid(), proxies=proxies, scraping_bee=scraping_bee)
        data["entries"] = self.__add_sub_articles(data["entries"])
        return data

    def topic_headlines(self, topic: str, proxies: Optional[dict] = None, scraping_bee: Optional[str] = None) -> Dict[str, list]:
        valid_topics = {"WORLD", "NATION", "BUSINESS", "TECHNOLOGY", "ENTERTAINMENT", "SCIENCE", "SPORTS", "HEALTH"}
        if topic.upper() in valid_topics:
            data = self.__parse_feed(
                f"{self.BASE_URL}/headlines/section/topic/{topic.upper()}{self.__ceid()}",
                proxies=proxies,
                scraping_bee=scraping_bee,
            )
        else:
            data = self.__parse_feed(
                f"{self.BASE_URL}/topics/{topic}{self.__ceid()}",
                proxies=proxies,
                scraping_bee=scraping_bee,
            )
        data["entries"] = self.__add_sub_articles(data["entries"])
        if data["entries"]:
            return data
        raise RuntimeError("unsupported topic")

    def geo_headlines(self, geo: str, proxies: Optional[dict] = None, scraping_bee: Optional[str] = None) -> Dict[str, list]:
        data = self.__parse_feed(
            f"{self.BASE_URL}/headlines/section/geo/{geo}{self.__ceid()}",
            proxies=proxies,
            scraping_bee=scraping_bee,
        )
        data["entries"] = self.__add_sub_articles(data["entries"])
        return data

    def search(
        self,
        query: str,
        helper: bool = True,
        when: Optional[str] = None,
        from_: Optional[str] = None,
        to_: Optional[str] = None,
        proxies: Optional[dict] = None,
        scraping_bee: Optional[str] = None,
    ) -> Dict[str, list]:
        if when:
            query += f" when:{when}"

        if from_ and not when:
            query += f" after:{self.__from_to_helper(validate=from_)}"
        if to_ and not when:
            query += f" before:{self.__from_to_helper(validate=to_)}"

        if helper:
            query = self.__search_helper(query)

        search_ceid = self.__ceid().replace("?", "&")
        data = self.__parse_feed(
            f"{self.BASE_URL}/search?q={query}{search_ceid}",
            proxies=proxies,
            scraping_bee=scraping_bee,
        )
        data["entries"] = self.__add_sub_articles(data["entries"])
        return data


def get_news(search: str, start_date: dt.date = dt.date(2020, 1, 1), end_date: dt.date = dt.date(2023, 9, 1), step_days: int = 5) -> List[dict]:
    """Replicates the ``get_news`` helper from the notebook."""

    gn = GoogleNews(lang="en")
    stories: List[dict] = []
    delta = dt.timedelta(days=step_days)
    date_list = pd.date_range(start_date, end_date).tolist()

    for date in date_list[:-1]:
        result = gn.search(search, from_=(date).strftime("%Y-%m-%d"), to_=(date + delta).strftime("%Y-%m-%d"))
        newsitem = result["entries"]
        for item in newsitem:
            stories.append({
                "title": item.title,
                "link": item.link,
                "published": item.published,
            })
    return stories


# Utility loaders -------------------------------------------------------------
def load_news_dataset(path: Path = NEWS_DATA_PATH) -> pd.DataFrame:
    """Load ``News_dataset.csv`` using the same column assumptions as the notebook."""

    if not path.exists():  # pragma: no cover - mirrors notebook expectation
        raise FileNotFoundError(f"News dataset not found at {path}")

    names = ["Title", "link", "Date", "Ex.", "Company"]
    df = pd.read_csv(path, names=names, header=None, skiprows=1)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def _transformer_device() -> int:
    """Return ``0`` if CUDA is available otherwise ``-1`` (CPU)."""

    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


def _safe_concat(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# Main ML class ---------------------------------------------------------------
class ML_build:
    """Core class replicating the behaviour of the original notebook."""

    def __init__(self) -> None:
        self.df_news: Optional[pd.DataFrame] = None
        self.df_stock: Optional[pd.DataFrame] = None
        self.Adjustment: Dict[str, float] = {}
        self.model_run_final: Optional[pd.DataFrame] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.LR_cv_results: Optional[np.ndarray] = None
        self.RF_cv_results: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Data ingestion & preparation
    # ------------------------------------------------------------------
    def news_df(self, df: pd.DataFrame) -> None:
        self.df_news = df.copy()
        self.df_news.loc[:, "Date"] = pd.to_datetime(self.df_news["Date"])

    def data_preparation(self, title_column: str = "Title", drop_dup: bool = True, sources_sep: bool = True, Non_eng: str = "drop") -> None:
        if self.df_news is None:
            raise RuntimeError("News dataframe is not initialised. Call news_df first.")

        if drop_dup:
            self.df_news = self.df_news.drop_duplicates()
        else:
            print("duplicates were not excluded")

        if Non_eng == "drop":
            self.df_news = self.df_news[self.df_news[title_column].apply(str.isascii) == True]  # noqa: E712
        else:
            print("other languages were not excluded")

        if "Source" not in self.df_news.columns:
            self.df_news[["Title", "Source"]] = self.df_news["Title"].str.rsplit(" - ", n=1, expand=True)

    def first_story_flag(self, Light_model: bool = True) -> None:
        if self.df_news is None:
            raise RuntimeError("News dataframe is not initialised. Call news_df first.")

        if not Light_model:
            import spacy

            sent_vecs: Dict[str, np.ndarray] = {}
            docs = []
            spacy.cli.download("en_core_web_lg")
            nlp = spacy.load("en_core_web_lg")
            for title in tqdm(self.df_news.Title.head(1000)):
                doc = nlp(title)
                docs.append(doc)
                sent_vecs.update({title: doc.vector})
            sentences = list(sent_vecs.keys())
            vectors = list(sent_vecs.values())
            x = np.array(vectors)
            n_classes: Dict[float, int] = {}
            for i in tqdm(np.arange(0.001, 1, 0.002)):
                dbscan = DBSCAN(eps=i, min_samples=2, metric="cosine").fit(x)
                n_classes.update({i: len(pd.Series(dbscan.labels_).value_counts())})
            optimal_eps = max(n_classes, key=n_classes.get)
            dbscan = DBSCAN(eps=optimal_eps, min_samples=2, metric="cosine").fit(x)
            result_dbscan = pd.DataFrame({"DBSCAN": dbscan.labels_, "sent": self.df_news.Title})
            self.df_news = self.df_news.merge(result_dbscan, left_on="Title", right_on="sent")
        else:
            vec = TfidfVectorizer(stop_words="english")
            x = vec.fit_transform(self.df_news.head(1000).Title)
            n_classes: Dict[float, int] = {}
            for i in tqdm(np.arange(0.001, 1, 0.002)):
                dbscan = DBSCAN(eps=i, min_samples=2, metric="cosine").fit(x)
                n_classes.update({i: len(pd.Series(dbscan.labels_).value_counts())})

            final_frames: List[pd.DataFrame] = []
            for company in tqdm(self.df_news["Company"].unique()):
                df_train = self.df_news[self.df_news["Company"] == company]
                vec = TfidfVectorizer(stop_words="english")
                x = vec.fit_transform(df_train.Title)
                optimal_eps = max(n_classes, key=n_classes.get)
                dbscan = DBSCAN(eps=optimal_eps, min_samples=2, metric="cosine").fit(x)
                result_dbscan = pd.DataFrame({"DBSCAN": dbscan.labels_, "sent": df_train.Title})
                final_frames.append(result_dbscan)

            final_dbscan_df = _safe_concat(final_frames)
            self.df_news = self.df_news.merge(final_dbscan_df, left_on="Title", right_on="sent", how="left")

        story_list: List[int] = []

        def first_occurrence(value: int) -> int:
            if value in story_list:
                return 0
            story_list.append(value)
            return 1

        self.df_news["First_story"] = self.df_news["DBSCAN"].apply(first_occurrence)

    # ------------------------------------------------------------------
    # BERT feature engineering
    # ------------------------------------------------------------------
    def BERT_sentiment_test(self, column: str = "Title") -> None:
        if self.df_news is None:
            raise RuntimeError("News dataframe is not initialised. Call news_df first.")

        from datasets import Dataset
        from transformers import pipeline
        from transformers.pipelines.pt_utils import KeyDataset

        dataset = Dataset.from_pandas(self.df_news.reset_index(drop=True))
        nlp = pipeline(
            task="sentiment-analysis",
            model="ProsusAI/finbert",
            device=_transformer_device(),
            batch_size=100,
        )

        result: List[dict] = []
        for out in tqdm(nlp(KeyDataset(dataset, column))):
            result.append(out)
        result_df = pd.DataFrame(result)
        if {"label", "score"}.issubset(result_df.columns):
            result_df = result_df.rename(columns={"label": "FinBERT", "score": "Fin_BERT_score"})
        self.df_news = self.df_news.reset_index(drop=True).join(result_df.set_axis(self.df_news.index))

    def BERT_quality_test(self, column: str = "Title") -> None:
        if self.df_news is None:
            raise RuntimeError("News dataframe is not initialised. Call news_df first.")

        from datasets import Dataset
        from transformers import pipeline
        from transformers.pipelines.pt_utils import KeyDataset

        dataset = Dataset.from_pandas(self.df_news.reset_index(drop=True))
        nlp = pipeline(
            "text-classification",
            model="ikoghoemmanuell/finetuned_fake_news_roberta",
            device=_transformer_device(),
            batch_size=100,
        )

        result: List[dict] = []
        for out in tqdm(nlp(KeyDataset(dataset, column))):
            result.append(out)
        result_df = pd.DataFrame(result)
        if {"label", "score"}.issubset(result_df.columns):
            result_df = result_df.rename(columns={"label": "FakeBERT", "score": "FakeBERT_score"})
        self.df_news = self.df_news.reset_index(drop=True).join(result_df.set_axis(self.df_news.index))

    def BERT_NER_test(self, column: str = "Title") -> None:
        if self.df_news is None:
            raise RuntimeError("News dataframe is not initialised. Call news_df first.")

        from datasets import Dataset
        from transformers import pipeline
        from transformers.pipelines.pt_utils import KeyDataset

        dataset = Dataset.from_pandas(self.df_news.reset_index(drop=True))
        nlp = pipeline(
            task="ner",
            model="dslim/bert-base-NER",
            device=_transformer_device(),
            batch_size=100,
        )

        result: List[dict] = []
        for out in tqdm(nlp(KeyDataset(dataset, column))):
            tokens = [token["word"] for token in out]
            entities = [token["entity"] for token in out]
            result.append(dict.fromkeys(entities, tokens))
        result_df = pd.DataFrame(result)
        self.df_news = self.df_news.reset_index(drop=True).join(result_df.set_axis(self.df_news.index))

    def save_BERT(self, path: Path = BERT_RESULTS_PATH) -> None:
        if self.df_news is None:
            raise RuntimeError("News dataframe is not initialised. Call news_df first.")
        self.df_news.to_csv(path, index=False)

    def load_BERT(self, path: Path = BERT_RESULTS_PATH, nrows: Optional[int] = None) -> None:
        if not path.exists():
            raise FileNotFoundError(f"BERT results not found at {path}. Run save_BERT first.")
        self.df_news = pd.read_csv(path, low_memory=False, nrows=nrows)
        self.df_news["Date"] = pd.to_datetime(self.df_news["Date"])

    # ------------------------------------------------------------------
    # Date adjustments
    # ------------------------------------------------------------------
    def time_publication_adjustment(self, naive: bool) -> None:
        if self.df_news is None:
            raise RuntimeError("News dataframe is not initialised. Call news_df first.")

        self.df_news["Date"] = pd.to_datetime(self.df_news["Date"])

        if naive:
            def offset_date(start: pd.Timestamp, offset: int) -> pd.Timestamp:
                return start + pd.offsets.CustomBusinessDay(n=offset, calendar=USFederalHolidayCalendar())

            offset = 0
            self.df_news["Date"] = self.df_news["Date"].apply(lambda x: offset_date(x, offset))
            self.df_news["Date"] = self.df_news["Date"].dt.date
            self.df_news["Source"] = self.df_news["Title"].str.split(" - ").str[1]
            return

        from sklearn.model_selection import GridSearchCV  # noqa: F401  # kept to match notebook imports
        self.Adjustment = {}
        for source in self.df_news["Source"].unique():
            df_train = self.df_news[self.df_news["Source"] == source].copy()

            def objective(trial):
                lag = trial.suggest_categorical("lag", [0, 0.25, 0.5, 0.75])

                def offset_date(df_row, offset: int = 0, offset2: float = lag * 4):
                    try:
                        first = df_row["Date"]
                        second = pd.offsets.CustomBusinessDay(n=offset, calendar=USFederalHolidayCalendar())
                        third = pd.offsets.DateOffset(day=offset2)
                        return first + second - third
                    except Exception:
                        return df_row["Date"] + pd.offsets.CustomBusinessDay(n=offset, calendar=USFederalHolidayCalendar())

                df_temp = df_train.copy()
                df_temp["Date"] = df_temp.apply(lambda x: offset_date(x), axis=1)
                df_temp["Date"] = df_temp["Date"].dt.date

                def get_current_prediction(df_sources):
                    new_class = ML_build()
                    new_class.news_df(df_sources)
                    new_class.One_hot_encode()
                    new_class.import_stock_data()
                    new_class.prepare_final()
                    new_class.train_test_split()
                    model = RandomForestClassifier().fit(new_class.X_train, new_class.y_train)
                    y_pred = model.predict(new_class.X_train)
                    return f1_score(new_class.y_train, y_pred)

                return get_current_prediction(df_temp)

            sampler = TPESampler(seed=42)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(objective, n_trials=5)
            self.Adjustment[source] = study.best_trial.params["lag"]

    # ------------------------------------------------------------------
    # Stock market data
    # ------------------------------------------------------------------
    def import_stock_data(self, period: str = "4y") -> None:
        if self.df_news is None:
            raise RuntimeError("News dataframe is not initialised. Call news_df first.")

        companies = self.df_news["Company"].dropna().unique().tolist()
        if not companies:
            raise RuntimeError("No companies found in news dataframe.")

        sp500 = yf.Ticker("^GSPC").history(period=period)
        sp500["Returns"] = (sp500.Close - sp500.Open) / sp500.Open
        sp500 = sp500.reset_index()
        sp500["Date"] = pd.to_datetime(sp500["Date"]).dt.date
        sp500 = sp500[["Date", "Returns"]].rename(columns={"Returns": "SP500_returns"})
        sp500["SP500_returns_yesterday"] = sp500["SP500_returns"].shift(-1)

        frames: List[pd.DataFrame] = []
        for comp in companies:
            ticker = yf.Ticker(comp)
            hist = ticker.history(period=period)
            hist = hist.reset_index()
            hist["Date"] = pd.to_datetime(hist["Date"]).dt.date
            hist["symbol"] = comp
            hist = hist.merge(sp500, on="Date", how="left")
            hist["Returns"] = (hist.Close - hist.Open) / hist.Open
            hist["Returns_binary"] = 0
            hist.loc[hist["Returns"] > 0, "Returns_binary"] = 1
            hist["Returns_yesterday"] = hist["Returns"].shift(-1)
            hist["surprise_percent"] = self._get_surprise_percent(ticker, hist["Date"])
            hist["market_cap"] = self._estimate_market_cap(ticker, hist["Close"])
            frames.append(hist)

        stock_data = _safe_concat(frames)
        stock_data = stock_data.merge(sp500, on="Date", how="left", suffixes=("", "_index"))
        stock_data = stock_data.rename(columns={"SP500_returns_index": "SP500_returns"})
        self.df_stock = stock_data

    def _estimate_market_cap(self, ticker: yf.Ticker, close: pd.Series) -> pd.Series:
        try:
            shares = ticker.get_shares_full(start="2019-01-01").iloc[-1]
            if isinstance(shares, pd.Series):
                shares = shares.iloc[0]
        except Exception:
            shares = None
        if shares is None:
            try:
                shares = ticker.info.get("sharesOutstanding")
            except Exception:
                shares = None
        if shares is None:
            return pd.Series(np.nan, index=close.index)
        return close * shares

    def _get_surprise_percent(self, ticker: yf.Ticker, dates: pd.Series) -> pd.Series:
        try:
            earnings = ticker.get_earnings_dates(limit=64).reset_index()
        except Exception:
            earnings = pd.DataFrame(columns=["index", "Surprise(%)"])
        if earnings.empty:
            return pd.Series(np.nan, index=dates.index)
        surprise_col = None
        for col in earnings.columns:
            if "surprise" in col.lower():
                surprise_col = col
                break
        if surprise_col is None:
            return pd.Series(np.nan, index=dates.index)
        earnings = earnings.rename(columns={"index": "Date", surprise_col: "surprise_percent"})
        earnings["Date"] = pd.to_datetime(earnings["Date"]).dt.date
        merger = pd.DataFrame({"Date": dates})
        merged = merger.merge(earnings[["Date", "surprise_percent"]], on="Date", how="left")
        return merged["surprise_percent"]

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def One_hot_encode(self, Interactions: bool = True, OHE: bool = True) -> None:
        if self.df_news is None:
            raise RuntimeError("News dataframe is not initialised. Call news_df first.")

        self.df_news = self.df_news.reset_index(drop=True)
        threshold = 10
        value_counts = self.df_news.Source.value_counts()
        to_remove = value_counts[value_counts <= threshold].index
        self.df_news.loc[:, "Source"] = self.df_news.Source.replace(to_remove, "Other")

        columns_ohe = ["FinBERT", "FakeBERT"]

        if OHE:
            ohe = OneHotEncoder(categories="auto")
            feature_arr = ohe.fit_transform(self.df_news[columns_ohe]).toarray()
            feature_labels = [item for sublist in ohe.categories_ for item in sublist]
            features = pd.DataFrame(feature_arr, columns=feature_labels)
            self.df_news = pd.concat([self.df_news, features], axis=1)
        else:
            cols1 = ["negative", "neutral", "positive"]
            for col in cols1:
                self.df_news[col] = 0
                self.df_news.loc[self.df_news["FinBERT"] == col, col] = 1
            cols2 = ["LABEL_0", "LABEL_1"]
            for col in cols2:
                self.df_news[col] = 0
                self.df_news.loc[self.df_news["FakeBERT"] == col, col] = 1

        if Interactions:
            self.df_news["positive*Label_1"] = self.df_news["positive"] * self.df_news["LABEL_1"]
            self.df_news["negative*Label_1"] = self.df_news["negative"] * self.df_news["LABEL_1"]

    def prepare_final(self, Interactions: bool = True) -> None:
        if self.df_news is None or self.df_stock is None:
            raise RuntimeError("News and stock data must be prepared before calling prepare_final.")

        self.df_news.loc[:, "Date"] = pd.to_datetime(self.df_news["Date"]).dt.date
        new_indicators = self.df_news.groupby(["Company", "Date"]).sum(numeric_only=True)
        df = pd.merge(
            self.df_stock,
            new_indicators,
            left_on=["symbol", "Date"],
            right_on=["Company", "Date"],
            how="inner",
        )
        df = df.set_index("Date")

        for date in df.index.unique():
            subset = df.loc[df.index == date]
            positive_sum = subset["positive"].sum()
            negative_sum = subset["negative"].sum()
            try:
                uncert = max(0, 1 - negative_sum / (positive_sum + negative_sum))
            except Exception:
                uncert = 0
            df.loc[df.index == date, "Uncert"] = uncert

        if Interactions:
            df["positive*Label_1*Uncert"] = df["positive*Label_1"] * df["Uncert"]
            df["negative*Label_1*Uncert"] = df["negative*Label_1"] * df["Uncert"]

        drop_cols = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Dividends",
            "Stock Splits",
            "Returns",
            "Unnamed: 0",
            "index",
            "symbol",
        ]
        drop_cols = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=drop_cols)
        self.df = df

    def Beta_calculation(self) -> None:
        if self.df_stock is None:
            raise RuntimeError("Stock dataframe is not initialised. Call import_stock_data first.")

        import statsmodels.api as sm

        frames: List[pd.DataFrame] = []
        for symbol, group in self.df_stock.groupby("symbol"):
            group = group.dropna(subset=["Returns", "SP500_returns"])
            if group.empty:
                continue
            X = sm.add_constant(group["SP500_returns"])
            model = sm.OLS(group["Returns"], X).fit()
            expected = model.predict(X)
            group = group.copy()
            group["beta"] = model.params.get("SP500_returns", np.nan)
            group["alpha"] = model.params.get("const", np.nan)
            group["expected_return"] = expected
            group["ar"] = group["Returns"] - expected
            frames.append(group)
        if frames:
            self.df_stock = pd.concat(frames, ignore_index=True)
        else:
            self.df_stock["ar"] = np.nan

    def prepare_news_df(self) -> pd.DataFrame:
        if self.df_stock is None:
            raise RuntimeError("Stock dataframe is not initialised. Call import_stock_data first.")
        if self.df_news is None:
            raise RuntimeError("News dataframe is not initialised. Call news_df first.")

        df_news = self.df_news.copy()
        df_news["Date"] = pd.to_datetime(df_news["Date"]).dt.date
        aggregations = {
            "positive": "sum",
            "negative": "sum",
            "neutral": "sum",
            "LABEL_0": "sum",
            "LABEL_1": "sum",
            "positive*Label_1": "sum",
            "negative*Label_1": "sum",
        }
        grouped = df_news.groupby(["Company", "Date"]).agg(aggregations).reset_index()
        grouped = grouped.rename(columns={
            "positive*Label_1": "positive_label_1",
            "negative*Label_1": "negative_label_1",
        })
        grouped["positive_label_0"] = grouped["positive"] - grouped["positive_label_1"]
        grouped["negative_label_0"] = grouped["negative"] - grouped["negative_label_1"]
        total = grouped[["positive", "negative", "neutral"]].sum(axis=1).replace(0, np.nan)
        grouped["rel_pos_label_1"] = grouped["positive_label_1"] / total
        grouped["rel_neg_label_1"] = grouped["negative_label_1"] / total
        grouped["rel_pos_label_0"] = grouped["positive_label_0"] / total
        grouped["rel_neg_label_0"] = grouped["negative_label_0"] / total
        grouped = grouped.fillna(0.0)
        grouped = grouped.drop(columns=["positive", "negative", "neutral"])

        stock = self.df_stock.copy()
        stock["Date"] = pd.to_datetime(stock["Date"]).dt.date

        merged = stock.merge(grouped, left_on=["symbol", "Date"], right_on=["Company", "Date"], how="left")
        merged = merged.rename(
            columns={
                "Dividends": "dividends",
                "Volume": "volume",
            }
        )
        if "ar" not in merged.columns:
            merged["ar"] = merged["Returns"] - merged["SP500_returns"]
        merged["const"] = 1.0
        keep = [
            "symbol",
            "Date",
            "ar",
            "const",
            "dividends",
            "volume",
            "market_cap",
            "surprise_percent",
            "positive_label_1",
            "negative_label_1",
            "positive_label_0",
            "negative_label_0",
            "rel_pos_label_1",
            "rel_neg_label_1",
            "rel_pos_label_0",
            "rel_neg_label_0",
        ]
        available = [col for col in keep if col in merged.columns]
        merged = merged[available]
        merged = merged.dropna(subset=["ar"])
        merged = merged.fillna(0.0)
        merged = merged.set_index(["symbol", "Date"]).sort_index()
        self.model_run_final = merged
        return merged

    # ------------------------------------------------------------------
    # Modelling utilities
    # ------------------------------------------------------------------
    def train_test_split(self, test_size: float = 0.33, random_state: int = 42) -> None:
        if not hasattr(self, "df"):
            raise RuntimeError("Call prepare_final before train_test_split.")
        X = self.df.drop(["Returns_binary"], axis=1).copy()
        y = self.df["Returns_binary"].copy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

    def LR_class(self, naive: bool = True, metric: str = "f1_weighted") -> None:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Run train_test_split before LR_class.")

        from sklearn import preprocessing

        scaler = preprocessing.StandardScaler().fit(self.X_train)
        X_scaled = scaler.transform(self.X_train)
        self.X_train = X_scaled

        if naive:
            lr_final = LogisticRegression()
            scores = cross_val_score(lr_final, self.X_train, self.y_train, cv=5, scoring=metric)
            self.LR_cv_results = scores
            print("Results of basic Logistic Regression:")
            print("Metric: ", metric, " mean score of cross val: ", scores.mean())
        else:
            def objective(trial):
                params = {
                    "tol": trial.suggest_float("tol", 1e-6, 1e-3),
                    "C": trial.suggest_float("C", 1e-2, 1, log=True),
                }
                model = LogisticRegression(**params)
                model.fit(self.X_train, self.y_train)
                scores_inner = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring="f1_weighted")
                return scores_inner.mean()

            sampler = TPESampler(seed=42)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(objective, n_trials=40)
            params_final = study.best_trial.params
            lr_final = LogisticRegression(**params_final)
            scores = cross_val_score(lr_final, self.X_train, self.y_train, cv=5, scoring="f1_weighted")
            self.LR_cv_results = scores
            print("Results of hyper-tuned Logistic Regression:")
            print("Metric: ", metric, " mean score of cross val: ", scores.mean())

    def RF_class(self, naive: bool = True, metric: str = "f1_weighted") -> None:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Run train_test_split before RF_class.")

        if naive:
            rf_final = RandomForestClassifier()
            scores = cross_val_score(rf_final, self.X_train, self.y_train, cv=5, scoring=metric)
            self.RF_cv_results = scores
            print("Results of basic Random Forest:")
            print("Metric: ", metric, " mean score of cross val: ", scores.mean())
        else:
            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("tol", 5, 200),
                }
                model = RandomForestClassifier(**params)
                model.fit(self.X_train, self.y_train)
                scores_inner = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring="f1_weighted")
                return scores_inner.mean()

            sampler = TPESampler(seed=123)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(objective, n_trials=20)
            params_final = study.best_trial.params
            rf_final = RandomForestClassifier(**params_final)
            scores = cross_val_score(rf_final, self.X_train, self.y_train, cv=5, scoring="f1_weighted")
            self.RF_cv_results = scores
            print("Results of hyper-tuned Random Forest:")
            print("Metric: ", metric, " mean score of cross val: ", scores.mean())
