"""Data parser module for collecting fake news training data."""

from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd
import requests


class DataParser:
    """Collects news articles for training a fake news classifier."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("NEWSAPI_KEY")

    def fetch_news(self, keywords: List[str], *, from_date: Optional[str] = None,
                   to_date: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
        """Fetch articles from the NewsAPI service.

        Parameters
        ----------
        keywords:
            List of search keywords.
        from_date, to_date:
            Optional ISO date strings to bound the search.
        limit:
            Maximum number of articles to return.

        Returns
        -------
        pandas.DataFrame
            Data frame containing the retrieved articles with a ``text`` column.
        """
        if self.api_key is None:
            raise ValueError("NewsAPI key required to fetch articles")

        url = "https://newsapi.org/v2/everything"
        query = " OR ".join(keywords)
        params = {
            "q": query,
            "pageSize": limit,
            "apiKey": self.api_key,
        }
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json().get("articles", [])

        records = [
            {
                "text": f"{article.get('title', '')} {article.get('description', '')}",
                "source": article.get("source", {}).get("name"),
                "published_at": article.get("publishedAt"),
            }
            for article in data
        ]
        return pd.DataFrame(records)

    def load_dataset(self, path: str) -> pd.DataFrame:
        """Load an already prepared data set from disk."""
        return pd.read_csv(path)
