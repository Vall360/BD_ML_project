"""Lightweight pipeline to run panel regressions with precomputed news features."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sci
import yfinance as yf
from stargazer.stargazer import Stargazer


BASE_DIR = Path(__file__).resolve().parent
BERT_OHE_PATH = BASE_DIR / "BERT_project_OHE.csv"
SOURCES_CLASS_PATH = BASE_DIR / "Sources_class.xlsx"
DF_STOCK_PATH = BASE_DIR / "df_stock.csv"
DF_STOCK_PATH_EXCEL = BASE_DIR / "df_stock.xlsx"
REPORT_PATH = BASE_DIR / "Report0.html"
ORIGINAL_PANEL = False

def import_stock_data(news_df: pd.DataFrame) -> pd.DataFrame:
  """Download stock data for tickers present in the news dataframe."""

  companies = news_df["Company"].dropna().unique().tolist()
  stock_data = pd.DataFrame()

  sp500 = yf.Ticker('^GSPC').history(period="5y")
  sp500 = sp500.reset_index()
  sp500.loc[:, "Date"] = pd.to_datetime(sp500.loc[:, "Date"]).dt.date
  sp500['SP500_returns'] = (sp500['Close'] - sp500['Open']) / sp500['Open']
  sp500['SP500_returns_yesterday'] = sp500['SP500_returns'].shift(-1)
  sp500 = sp500[['Date', 'SP500_returns', 'SP500_returns_yesterday']]

  vix = yf.Ticker('^VIX').history(period="5y")
  vix = vix.reset_index()
  vix.loc[:, "Date"] = pd.to_datetime(vix.loc[:, "Date"]).dt.date
  vix['VIX_close'] = vix['Close']
  vix['VIX_returns'] = vix['VIX_close'].pct_change()
  vix['VIX_returns_yesterday'] = vix['VIX_returns'].shift(-1)
  vix = vix[['Date', 'VIX_close', 'VIX_returns', 'VIX_returns_yesterday']]

  for comp in companies:
    try:
      ticker = yf.Ticker(comp)
      hist = ticker.history(period="5y")

      hist = hist.reset_index()
      hist.loc[:, 'Date'] = pd.to_datetime(hist.loc[:, 'Date'])
      hist['Date'] = hist['Date'].dt.date
      hist['Date'] = pd.to_datetime(hist['Date'])

      stock_count = pd.DataFrame(ticker.financials)
      stock_count = stock_count.T
      if 'Diluted Average Shares' in stock_count.columns:
        stock_count = stock_count['Diluted Average Shares'].sort_index()
        stock_count = pd.DataFrame(stock_count)

        tol = pd.Timedelta('7 day')
        hist = pd.merge_asof(
            hist.sort_values('Date'),
            stock_count,
            right_index=True,
            left_on='Date',
            direction='nearest',
            tolerance=tol)
        hist['Diluted Average Shares'].fillna(method='ffill', inplace=True)
        hist['Diluted Average Shares'].fillna(method='bfill', inplace=True)
        hist["Market_Cap"] = hist["Diluted Average Shares"] * hist["Close"]
      else:
        hist["Market_Cap"] = np.nan

      hist['Date'] = hist['Date'].dt.date

      try:
        earn_releases = ticker.get_earnings_dates(limit=25)
        earn_releases = earn_releases.reset_index()
        earn_releases.rename(columns={'Earnings Date': 'Date'}, inplace=True)
        earn_releases.loc[:, 'Date'] = pd.to_datetime(earn_releases.loc[:, 'Date'])
        earn_releases['Date'] = earn_releases['Date'].dt.date
        hist = pd.merge(hist, earn_releases, how="left", on='Date')
      except Exception as exc:  # pragma: no cover - best effort logging
        print(f"Warning: earnings fetch failed for {comp}: {exc}")

      hist = hist.merge(sp500, on='Date', how='left')
      hist = hist.merge(vix, on='Date', how='left')
      hist['symbol'] = comp

      stock_data = pd.concat([stock_data, hist], ignore_index=True)
    except Exception as exc:  # pragma: no cover - best effort logging
      print(f"Warning: failed to download data for {comp}: {exc}")

  stock_data["Returns"] = (stock_data.Close - stock_data.Open) / stock_data.Open
  stock_data["Returns_binary"] = (stock_data["Returns"] > 0).astype(int)
  stock_data["Returns_yesterday"] = stock_data["Returns"].shift(-1)
  return stock_data


def beta_calculation(df: pd.DataFrame, window: int, stock: str = 'Returns',
                     market: str = 'SP500_returns') -> tuple[pd.DataFrame, pd.DataFrame]:
  """Compute firm-level betas and abnormal returns around earnings events."""

  if 'symbol' not in df.columns:
    raise KeyError("DataFrame must contain 'symbol' column for beta calculation.")

  symbols = df['symbol'].dropna().unique()
  df = df.set_index(['symbol', 'Date'])

  df['Beta'] = np.nan
  df['AR'] = np.nan
  df['AR_signif'] = np.nan
  df['Event'] = np.nan
  df['Event_ID'] = np.nan

  est_window = 50
  event_window = 10

  test_df = df.copy()
  risk_free = yf.Ticker('^IRX').history(period="5y")['Close'] * 0.01
  risk_free = risk_free.reset_index()
  risk_free.loc[:, 'Date'] = pd.to_datetime(risk_free.loc[:, 'Date'])
  risk_free['Date'] = risk_free['Date'].dt.date

  if 'Surprise(%)' not in df.columns:
    print("Warning: 'Surprise(%)' column missing; beta calculation skipped.")
    df = df.reset_index()
    return df, test_df.reset_index()

  unique_id = 0
  for symbol in symbols:
    unique_id = round(unique_id / 100) * 100
    unique_id += 100

    sliced_company = df.loc[symbol].copy().reset_index()

    event_indices = sliced_company.index[sliced_company['Surprise(%)'].notna()].tolist()
    for event in event_indices:
      unique_id += 1
      start = int(event - event_window / 2 - est_window)
      end = int(event + event_window / 2)

      sliced = sliced_company[start:end].copy()
      sliced['Event_ID'] = unique_id
      sliced.loc[int(event - event_window / 2): end, 'Event'] = 1

      stock_var = sliced.loc[start: int(event - event_window / 2), stock]
      market_var = sliced.loc[start: int(event - event_window / 2), market]

      covariance = stock_var.cov(market_var)
      variance = market_var.var()
      beta = covariance / variance if variance else np.nan
      sliced['Beta'] = beta

      sliced['model_ret'] = 0 + sliced['Beta'] * (sliced[market] - 0)
      sliced['AR'] = (sliced['Returns'] - sliced['model_ret']) * 100

      crit_val = sci.norm.ppf(0.95)

      def test_crit(stat: float, crit: float) -> int:
        if stat >= crit:
          return 1
        if stat <= -crit:
          return 2
        return 0

      sliced['AR_signif'] = sliced['AR'].apply(lambda x: test_crit(x, crit_val))
      sliced_company.update(sliced, overwrite=False)

    sliced_company['symbol'] = symbol
    sliced_company = sliced_company.set_index(['symbol', 'Date'])
    df.update(sliced_company, overwrite=False)

  df = df.reset_index()
  return df, test_df.reset_index()


def prepare_news_df(news_df: pd.DataFrame, curated: bool = True,
                    organisation: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Aggregate news metrics by company/date with optional filtering."""

  news_df = news_df.copy()
  news_df = news_df.reset_index(drop=True)
  news_df['Date'] = pd.to_datetime(news_df['Date'], utc=True, errors='coerce').dt.date

  news_df['Rel_pos'] = news_df['positive']
  news_df['Rel_neg'] = news_df['negative']
  news_df['positive*Label_0'] = news_df['positive'] * news_df['LABEL_0']
  news_df['negative*Label_0'] = news_df['negative'] * news_df['LABEL_0']
  news_df['Rel_pos*Label_1'] = news_df['Rel_pos'] * news_df['LABEL_1']
  news_df['Rel_neg*Label_1'] = news_df['Rel_neg'] * news_df['LABEL_1']
  news_df['Rel_pos*Label_0'] = news_df['Rel_pos'] * news_df['LABEL_0']
  news_df['Rel_neg*Label_0'] = news_df['Rel_neg'] * news_df['LABEL_0']

  original_news_df = news_df.copy()

  if organisation and 'B-ORG' in news_df.columns:
    if np.issubdtype(news_df['B-ORG'].dtype, np.number):
      news_df = news_df[news_df['B-ORG'] == 1]
    else:
      news_df = news_df[news_df['B-ORG'].notna()]

  if curated:
    if not SOURCES_CLASS_PATH.exists():
      raise FileNotFoundError(f"Sources_class.xlsx not found at {SOURCES_CLASS_PATH}")
    sources_df = pd.read_excel(SOURCES_CLASS_PATH, header=1).fillna(0)
    news_df = news_df.merge(sources_df, on="Source", how='left')
    news_df = news_df.drop(["Source", "count"], axis=1)
    news_df = news_df[(news_df['Relevant'] > 0) | (news_df['Traditional'] > 0)]

  news_df = news_df[["Date", "Company", 'positive', 'negative',
                     'positive*Label_1', 'negative*Label_1',
                     'positive*Label_0', 'negative*Label_0',
                     'Rel_pos*Label_1', 'Rel_neg*Label_1',
                     'Rel_pos*Label_0', 'Rel_neg*Label_0']]

  news_df = news_df.groupby(['Company', 'Date']).sum()
  return news_df, original_news_df


def merge_stock_news(news_df: pd.DataFrame, stock_df: pd.DataFrame,
                     only_rel: bool = True, no_events: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Merge engineered news features with stock data and build event windows."""

  df_stock = pd.merge(stock_df, news_df.reset_index(),
                      left_on=['symbol', 'Date'],
                      right_on=['Company', 'Date'],
                      how="left")

  print(df_stock)
  df_stock = df_stock[df_stock['Date'] >= pd.to_datetime('2020-01-01').date() ]
  df_stock = df_stock[df_stock['Date'] <= pd.to_datetime('2023-09-01').date() ]

  uncertainty_df = pd.DataFrame()
  for date in df_stock.Date.unique():
    subset = df_stock[df_stock.Date == date]
    pos_sum = subset['positive'].sum()
    neg_sum = subset['negative'].sum()
    try:
      uncert = max(0, 1 - neg_sum / (pos_sum + neg_sum))
    except ZeroDivisionError:
      uncert = 0
    df_stock.loc[df_stock.Date == date, 'Uncert'] = uncert
    uncertainty_df = pd.concat([uncertainty_df, pd.DataFrame({"Date": [date], "Uncert": [uncert]})])

  df_stock["Event_only_index"] = np.nan
  if {'Event', 'Event_ID'}.issubset(df_stock.columns):
    mask = (df_stock["Event"] == 1) & (df_stock["Event_ID"].notna())
    df_stock.loc[mask, "Event_only_index"] = df_stock.loc[mask, "Event_ID"]

  df_stock["Illiquidity"] = abs(df_stock["Returns"]) / (
      df_stock["High"] * df_stock["High"] * df_stock["Volume"] / 2)

  grouped = df_stock.drop("Date", axis=1).groupby('symbol')
  results = {'symbol': [], 'illiquidity': []}
  for ticker, grp in grouped:
    results['symbol'].append(ticker)
    results['illiquidity'].append(grp["Illiquidity"].mean())

  res_df = pd.DataFrame(results)
  quant = res_df["illiquidity"].quantile(0.75)
  res_df['Illiq'] = res_df['illiquidity'].ge(quant).astype(int)
  df_stock = df_stock.merge(res_df[['symbol', 'Illiq']], on='symbol', how='left')

  new_cols = ["PL1_", "NL1_", "PL0_", "NL0_", "RPL1_", "RNL1_", "RPL0_", "RNL0_"]
  old_cols = ['positive*Label_1', 'negative*Label_1',
              'positive*Label_0', 'negative*Label_0',
              'Rel_pos*Label_1', 'Rel_neg*Label_1',
              'Rel_pos*Label_0', 'Rel_neg*Label_0']
  for lag in [20, 40, 60, 120]:
    names = [f"{prefix}{lag}" for prefix in new_cols]
    df_stock[names] = df_stock[old_cols].rolling(window=lag).sum()

  df_stock.reset_index(inplace=True, drop=True)

  if no_events:
    df_stock["Date"] = pd.to_datetime(df_stock["Date"])
    cols = ["Date", 'Volume', 'Dividends', 'Stock Splits', 'Market_Cap',
            'SP500_returns', 'SP500_returns_yesterday', 'VIX_close',
            'VIX_returns', 'VIX_returns_yesterday', 'Returns', 'AR',
            'EPS Estimate', 'Reported EPS', 'Surprise(%)', 'symbol',
            'positive*Label_1', 'negative*Label_1', 'positive*Label_0',
            'negative*Label_0', 'Uncert', 'Event_only_index',
            'Rel_pos*Label_1', 'Rel_neg*Label_1', 'Rel_pos*Label_0',
            'Rel_neg*Label_0', 'Illiq'] + [f"{prefix}{lag}" for prefix in new_cols for lag in [20, 40, 60, 120]]
    model_run = df_stock[cols].groupby(['symbol', 'Date']).sum()
  else:
    cols = ['Date', 'Volume', 'Dividends', 'Stock Splits', 'Market_Cap',
            'SP500_returns', 'SP500_returns_yesterday', 'VIX_close',
            'VIX_returns', 'VIX_returns_yesterday', 'Returns', 'AR',
            'EPS Estimate', 'Reported EPS', 'Surprise(%)', 'symbol',
            'Event', 'Event_ID', 'positive*Label_1', 'negative*Label_1',
            'positive*Label_0', 'negative*Label_0', 'Uncert', 'Event_only_index',
            'Rel_pos*Label_1', 'Rel_neg*Label_1', 'Rel_pos*Label_0', 'Rel_neg*Label_0',
            'Illiq'] + [f"{prefix}{lag}" for prefix in new_cols for lag in [20, 40, 60, 120]]
    model_data = df_stock[cols]
    model_data = model_data.drop(model_data[model_data['Event_only_index'] == 0.0].index)
    data_cols = set(model_data.columns) - {"Date", 'symbol'}
    agg_dict = {**dict.fromkeys(data_cols, 'sum'), **{'Date': 'first'}}
    model_run = model_data.groupby(['symbol', 'Event_only_index']).agg(agg_dict)

    def get_uncertainty(row: pd.Series, unc_df: pd.DataFrame) -> float:
      value = unc_df.loc[unc_df['Date'] == row['Date'], 'Uncert']
      return value.iloc[0] if not value.empty else np.nan

    if 'Uncert' in model_run.columns:
      model_run.drop("Uncert", axis=1, inplace=True)
    model_run['Uncert'] = model_run.apply(lambda row: get_uncertainty(row, uncertainty_df), axis=1)

  def new_dummy(df: pd.DataFrame, key_col: str, cols_to_apply: list[str]) -> pd.DataFrame:
    for col in cols_to_apply:
      name = f"{col}_{key_col}"
      df[name] = df[key_col] * df[col]
    return df

  def new_index(df: pd.DataFrame, pos_cols: list[str], all_cols: list[str]) -> pd.DataFrame:
    for col in pos_cols:
      name = f"{col}_index"
      neg_col = all_cols[all_cols.index(col) + 1]
      df[name] = (df[col] - df[neg_col]) / (df[col] + df[neg_col]) / 10
    return df

  lag_list = ['PL1_20', 'NL1_20', 'PL0_20', 'NL0_20', 'RPL1_20', 'RNL1_20',
              'RPL0_20', 'RNL0_20', 'PL1_40', 'NL1_40', 'PL0_40', 'NL0_40',
              'RPL1_40', 'RNL1_40', 'RPL0_40', 'RNL0_40', 'PL1_60', 'NL1_60',
              'PL0_60', 'NL0_60', 'RPL1_60', 'RNL1_60', 'RPL0_60', 'RNL0_60',
              'PL1_120', 'NL1_120', 'PL0_120', 'NL0_120', 'RPL1_120', 'RNL1_120',
              'RPL0_120', 'RNL0_120']
  positive_values = ['PL1_20', 'PL0_20', 'RPL1_20', 'RPL0_20', 'PL1_40', 'PL0_40',
                     'RPL1_40', 'RPL0_40', 'PL1_60', 'PL0_60', 'RPL1_60', 'RPL0_60',
                     'PL1_120', 'PL0_120', 'RPL1_120', 'RPL0_120']

  model_run = new_index(model_run, positive_values, lag_list)

  index_list = [f"{col}_index" for col in positive_values]
  model_run = new_dummy(model_run, "Uncert", index_list)
  model_run = new_dummy(model_run, "Illiq", index_list)

  col_list = ['positive*Label_1', 'negative*Label_1', 'positive*Label_0',
              'negative*Label_0', 'Rel_pos*Label_1', 'Rel_neg*Label_1',
              'Rel_pos*Label_0', 'Rel_neg*Label_0']
  model_run = new_dummy(model_run, "Uncert", col_list)
  model_run = new_dummy(model_run, "Illiq", col_list)

  return model_run, df_stock


def run_regressions(model_run: tuple[pd.DataFrame, pd.DataFrame]) -> None:
  try:
    import linearmodels as lm
  except ImportError as exc:  # pragma: no cover - user action required
    raise ImportError("linearmodels is required. Install it with 'pip install linearmodels'.") from exc

  model_run_df = model_run[0].copy()
  model_run_df['const'] = 1
  model_run_df['Market_Cap'] = np.log(model_run_df['Market_Cap'])
  model_run_df['Volume'] = np.log(model_run_df['Volume'])
  model_run_final = model_run_df.copy()
  model_run_final.columns = (model_run_final.columns
                             .str.lower()
                             .str.replace('*', '_')
                             .str.replace('(%)', '_percent', regex=False))
  if ORIGINAL_PANEL:
    TARGET_OBSERVATIONS = 343
    SYMBOL_WHITELIST = (
      "QCOM", "ABT", "MS", "UPS", "CRM", "T", "NFLX", "VZ", "COST", "WFC",
      "PEP", "MCD", "ACN", "CMCSA", "NKE", "MRK", "INTC", "ORCL", "PYPL", "TMO",
      "SCHW", "LLY", "ADBE", "DHR", "TXN", "CVX", "ABBV", "AVGO", "JPM", "MA",
      "V", "JNJ", "DIS", "AMZN", "AAPL", "GOOG", "HD", "PG", "MSFT",
    )
    symbols = SYMBOL_WHITELIST

    print(model_run_final)
    model_run_final.reset_index(inplace=True)
    print(model_run_final)
    ## Works good, keep for now
    ##model_run_final[model_run_final['symbol'].isin(set(symbols))]

    model_run_final = model_run_final[model_run_final['symbol'].isin(set(symbols))]
    model_run_final.sort_values('date', inplace=True)

    if len(model_run_final) > TARGET_OBSERVATIONS:
      model_run_final = model_run_final.iloc[-TARGET_OBSERVATIONS:]
      cut_date = model_run_final['date'].min()
      print(f"Trimmed panel to {len(model_run_final)} rows starting from {cut_date}")
    else:
      print(f"Panel size {len(model_run_final)} rows; no trimming applied")

    model_run_final.set_index(['symbol', 'Event_only_index'], inplace=True)

  ORIGINAL_PANEL_V2 =True
  if ORIGINAL_PANEL_V2:
    dates = pd.read_excel("extracted_dates.xlsx", parse_dates=True)
    print(dates)

    from typing import Iterable, List, Tuple

    def _build_date_groups(ref_dates: Iterable[pd.Timestamp]) -> List[List[pd.Timestamp]]:
      groups: List[List[pd.Timestamp]] = []
      current: List[pd.Timestamp] = []
      prev: pd.Timestamp | None = None

      for raw_date in ref_dates:
        ts = pd.to_datetime(raw_date, errors="coerce")
        if pd.isna(ts):
          continue
        ts = ts.normalize()

        if prev is not None and ts < prev:
          if current:
            groups.append(current)
          current = [ts]
        else:
          current.append(ts)

        prev = ts

      if current:
        groups.append(current)

      return groups

    def flag_and_consume_dates(
        panel_df: pd.DataFrame,
        ref_dates: List[pd.Timestamp],
        ticker_col: str = "symbol",
        date_col: str = "date",
        flag_col: str = "ORIG_DATE",
    ) -> Tuple[pd.DataFrame, List[pd.Timestamp]]:
      if ticker_col not in panel_df.columns or date_col not in panel_df.columns:
        raise ValueError(f"panel_df must contain columns '{ticker_col}' and '{date_col}'.")

      pdf = panel_df.copy()
      pdf[date_col] = pd.to_datetime(pdf[date_col], errors="coerce")
      pdf[flag_col] = False
      valid_date_mask = pdf[date_col].notna()

      ref_groups = _build_date_groups(ref_dates)
      group_idx = 0

      ticker_sequence = list(dict.fromkeys(pdf[ticker_col].tolist()))

      for ticker in ticker_sequence:
        if group_idx >= len(ref_groups):
          break

        ticker_mask = (pdf[ticker_col] == ticker) & valid_date_mask
        if not ticker_mask.any():
          continue

        current_group = ref_groups[group_idx]
        group_dates = {d for d in current_group}

        ticker_dates = pdf.loc[ticker_mask, date_col].dt.normalize()
        match_mask = ticker_dates.isin(group_dates)

        if match_mask.any():
          pdf.loc[ticker_mask, flag_col] = match_mask.values
          group_idx += 1

      remaining = [dt for grp in ref_groups[group_idx:] for dt in grp]
      return pdf, remaining

    model_run_final, remaining_dates = flag_and_consume_dates(model_run_final.reset_index(), dates['date'].to_list())
    print(model_run_final, remaining_dates)


  model_run_final.to_excel('df_for_models_debug.xlsx', index=False)

  if ORIGINAL_PANEL_V2:
    model_run_final = pd.DataFrame(model_run_final)
    model_run_final = model_run_final[model_run_final['ORIG_DATE'] == True].copy()
    model_run_final = model_run_final.set_index(['symbol', 'Event_only_index'], inplace=True)

  formulas = [
      'ar ~ const + dividends + volume + market_cap + surprise_percent + positive_label_1 + negative_label_1 + EntityEffects',
      'ar ~ const + dividends + volume + market_cap + surprise_percent + positive_label_1 + negative_label_1 + positive_label_0 + EntityEffects',
      'ar ~ const + dividends + volume + market_cap + surprise_percent + positive_label_1 + negative_label_1 + negative_label_0 + EntityEffects',
      'ar ~ const + dividends + volume + market_cap + surprise_percent + positive_label_1 + negative_label_1 + positive_label_0 + negative_label_0 + EntityEffects',
      'ar ~ const + dividends + volume + market_cap + surprise_percent + rel_pos_label_1 + rel_neg_label_1 + EntityEffects',
      'ar ~ const + dividends + volume + market_cap + surprise_percent + rel_pos_label_1 + rel_neg_label_1 + rel_pos_label_0 + EntityEffects',
      'ar ~ const + dividends + volume + market_cap + surprise_percent + rel_pos_label_1 + rel_neg_label_1 + rel_neg_label_0 + EntityEffects',
      'ar ~ const + dividends + volume + market_cap + surprise_percent + rel_pos_label_1 + rel_neg_label_1 + rel_pos_label_0 + rel_neg_label_0 + EntityEffects'
  ]

  results = []
  for formula in formulas:
    model = lm.PanelOLS.from_formula(formula, data=model_run_final,
                                     check_rank=False, drop_absorbed=True)
    model_result = model.fit(cov_type='clustered', cluster_entity=True)
    print(model_result)
    results.append(model_result)

  report = Stargazer(results)
  REPORT_PATH.write_text(report.render_html())
  print(f"Panel regression report saved to {REPORT_PATH}")


def main() -> None:
  if not BERT_OHE_PATH.exists():
    raise FileNotFoundError(f"Pre-computed news file not found at {BERT_OHE_PATH}")

  news_df = pd.read_csv(BERT_OHE_PATH, low_memory=False)
  if 'Date' not in news_df.columns:
    raise KeyError("Expected 'Date' column in BERT_project_OHE.csv")

  if DF_STOCK_PATH.exists():
    print(f"Loading existing stock file from {DF_STOCK_PATH}")

    stock_df = pd.read_csv(DF_STOCK_PATH, low_memory=False)
    ##stock_df = import_stock_data(news_df)
    beta_df, _ = beta_calculation(stock_df, window=30)

  else:
    print("Stock file not found. Downloading data from Yahoo Finance...")
    stock_df = import_stock_data(news_df)
    stock_df.to_csv(DF_STOCK_PATH, index=False)
    stock_df.to_excel(DF_STOCK_PATH_EXCEL) ##reserve save to debug do not delete
    beta_df, _ = beta_calculation(stock_df, window=30)


  if 'Date' not in beta_df.columns:
    raise KeyError("Expected 'Date' column in df_stock.csv")

  beta_df['Date'] = pd.to_datetime(beta_df['Date'], errors='coerce').dt.date

  news_prepared, _ = prepare_news_df(news_df)
  model_run = merge_stock_news(news_prepared, beta_df, only_rel=False, no_events=False)
  run_regressions(model_run)


if __name__ == "__main__":
  main()
