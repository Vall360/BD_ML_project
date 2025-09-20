"""## paper clean"""
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import spacy
import tensorflow as tf
import torch
import yfinance as yf
from datasets import Dataset
import evaluate
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import BDay
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from stargazer.stargazer import Stargazer
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments

tqdm.pandas()

random_state = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
DRIVE_MIRROR_DIR = DATA_DIR / "colab_drive_mirror"
DRIVE_MIRROR_DIR.mkdir(parents=True, exist_ok=True)

NEWS_DATASET_PATH = DATA_DIR / "News_dataset.csv"
BERT_RESULTS_PATH = DATA_DIR / "BERT_project_results.csv"
BERT_DBSCAN_RESULTS_PATH = DATA_DIR / "BERT_DBSCAN_project_results.csv"
BERT_OHE_PATH = DATA_DIR / "BERT_project_OHE.csv"
SOURCES_CLASS_PATH = DATA_DIR / "Sources_class.xlsx"
DF_STOCK_PATH = DATA_DIR / "df_stock.csv"
RELEVANCE_MODEL_PATH = DATA_DIR / "Relevance_model"

USE_PRECOMPUTED_OHE = True

names = ["index", "Title", "link", "Date", "Ex.", "Company"]
try:
  News_dataset = pd.read_csv(NEWS_DATASET_PATH, header=0, names=names, sep=",")
except FileNotFoundError as exc:
  raise FileNotFoundError(f"News dataset not found at {NEWS_DATASET_PATH}") from exc


def mirror_to_drive(file_path: Path) -> None:
  destination = DRIVE_MIRROR_DIR / file_path.name
  try:
    shutil.copy(file_path, destination)
  except FileNotFoundError:
    print(f"Warning: source file {file_path} not found for mirroring.")
  except Exception as exc:  # pragma: no cover - best effort logging
    print(f"Warning: could not mirror {file_path} to {destination}: {exc}")

class ML_build:

  ## init class
  def __init__(self):
    return

  ## assign news dataframe into class
  def news_df(self, df):


    self.df_news = df.copy()
    self.df_news.loc[:,'Date'] = pd.to_datetime(self.df_news.loc[:,'Date'])
    return

  ## data cleaning from duplicates, other languages
  def data_preparation(self, title_column = 'Title',
                       drop_dup=True, sources_sep = True, Non_eng = 'drop'):
    ## Drop duplicates. Requiered step Given that we parsed for each company twice
    if drop_dup ==True:
      self.df_news = self.df_news.drop_duplicates()
    else:
      print("duplicates were not excluded")

    ## Drop Non-english news
    if Non_eng == 'drop':
      def isEnglish(s):
        return s.isascii()

      self.df_news = self.df_news[self.df_news[title_column].apply(isEnglish) == True]
    else:
      print("other languages were not excluded")


    if 'Source' not in self.df_news:
      self.df_news[['Title', 'Source']] = self.df_news['Title'].str.rsplit(' - ', n=1, expand=True)
    else:
      pass
    return


  ## finding important new by clusterisation based on rare words
  def first_story_flag(self, Light_model = False):
    if Light_model == False:
      ##advanced vectorisation. Does not support large df.
      sent_vecs = {}
      docs = []
      spacy.cli.download('en_core_web_lg')
      nlp = spacy.load('en_core_web_lg')
      for title in tqdm(self.df_news.Title.head(1000)):
        doc = nlp(title)
        docs.append(doc)
        sent_vecs.update({title: doc.vector})
        sentences = list(sent_vecs.keys())
        vectors = list(sent_vecs.values())
      x = np.array(vectors)
      ## search for optimal eps
      n_classes = {}
      for i in tqdm(np.arange(0.001, 1, 0.002)):
        dbscan = DBSCAN(eps=i, min_samples=2, metric='cosine').fit(x)
        n_classes.update({i: len(pd.Series (dbscan. labels_) .value_counts ())})
      ##print(n_classes)
      res_list = []
      for item in dbscan.labels_:
          if item not in res_list:
              res_list.append(item)
      ##print(res_list)


      ## Get most sensitive classification
      optimal_eps   = max(n_classes, key=n_classes.get)



      self.df_news["DBSCAN"] = -1
      self.df_news["First_story"] = 0
      self.df_news.set_index('Title',inplace=True)


      for i in self.df_news['Company'].unique():

        docs = []
        sent_vecs = {}
        titles = []


        ##for title in tqdm(self.df_news.loc[(self.df_news['Company'] == i), 'Title']):
        for title in tqdm(self.df_news.loc[(self.df_news['Company'] == i)].index.tolist()):
          titles.append(title)
          doc = nlp(title)
          docs.append(doc)
          sent_vecs.update({title: doc.vector})
          sentences = list(sent_vecs.keys())
          vectors = list(sent_vecs.values())
        x = np.array(vectors)

        edges = zip(*nx.to_edgelist(G))
        G1 = igraph.Graph(len(G), zip(*edges[:2]))
        D = 1 - np.array(G1.similarity_jaccard(loops=False))

        dbscan = DBSCAN(D, metric='precomputed', eps=optimal_eps, min_samples=2)

        ##dbscan = DBSCAN(eps=optimal_eps, min_samples=2, metric='cosine').fit(x)

        ##print(dbscan.labels_.shape, titles.shape)

        titles_dbscan = self.df_news.loc[(self.df_news['Company'] == i)].index.tolist()
        titles_dbscan = list(sent_vecs.keys())
        '''result_dbscan = pd.DataFrame({'DBSCAN': dbscan.labels_,
                                      'sent': titles_dbscan})
        if i == self.df_news['Company'].unique()[0]:
          self.df_news = self.df_news.merge(result_dbscan,
                                          left_on='Title', right_on='sent')
        else:
          self.df_news.loc[(self.df_news['Company'] == i), ]'''
        result_dbscan = pd.DataFrame({'DBSCAN': dbscan.labels_,
                                      'Title': titles_dbscan})
        result_dbscan.set_index('Title',inplace=True)

        Story_list = []
        def first_occurance(value):
          if value in Story_list:
            return 0
          else:
            Story_list.append(value)
            return 1

        result_dbscan['First_story'] = result_dbscan['DBSCAN'].apply(first_occurance)
        self.df_news.update(result_dbscan)
      return

      ##self.df_news = self.df_news.rename(columns={"A": "a", "B": "c"})


    else:
      ## run TF-ID model Based on http://ai.intelligentonlinetools.com/ml/text-clustering-word-embedding-machine-learning/
      from sklearn.feature_extraction.text import TfidfVectorizer

      vec = TfidfVectorizer( stop_words='english')##,
                      ##ngram_range=(1, 2))
      x = vec.fit_transform(self.df_news.head(1000).Title)
      ## search for optimal eps
      n_classes = {}
      for i in tqdm(np.arange(0.001, 1, 0.002)):
        dbscan = DBSCAN(eps=i, min_samples=2, metric='cosine').fit(x)
        n_classes.update({i: len(pd.Series (dbscan. labels_) .value_counts ())})
      print(n_classes)

      ## init full dataset

      Final_DBSCAN_df = pd.DataFrame()
      for company in tqdm(self.df_news['Company'].unique()):
        df_train = self.df_news[self.df_news['Company'] == company]
        vec = TfidfVectorizer( stop_words='english')##,
                        ##ngram_range=(1, 2))
        x = vec.fit_transform(self.df_news.Title)

        ## Get most sensetive classification
        optimal_eps   = max(n_classes, key=n_classes.get)
        dbscan = DBSCAN(eps=optimal_eps, min_samples=2, metric='cosine').fit(x)

        result_dbscan = pd.DataFrame({'DBSCAN': dbscan.labels_,
                                      'sent': self.df_news.Title})

        Final_DBSCAN_df = pd.concat([Final_DBSCAN_df, result_dbscan])
      ##print(Final_DBSCAN_df)
      self.df_news = self.df_news.merge(Final_DBSCAN_df,
                                        left_on='Title', right_on='sent')


    ## Get flag for first row in class
    Story_list = []
    def first_occurance(value):
      if value in Story_list:
        return 0
      else:
        Story_list.append(value)
        return 1

    self.df_news['First_story'] = self.df_news['DBSCAN'].apply(first_occurance)

    self.df_news.reset_index(inplace=True)


    return


  ## inital semtiment analysis


    return

  def BERT_sentiment_test(self, column = 'Title' ):
    from datasets import Dataset
    from transformers.pipelines.pt_utils import KeyDataset
    from transformers import pipeline

    nlp = pipeline(task='sentiment-analysis',
               model='ProsusAI/finbert', device=-1, batch_size=1000)


    result = []

    dataset = Dataset.from_pandas(self.df_news)

    for out in tqdm(nlp(KeyDataset(dataset, 'Title'))):
      result.append(out)
    print(out)
    result = pd.DataFrame(result)
    ##result.cou

    self.df_news= self.df_news.join(result.set_axis(self.df_news.index))

    ##self.df_news['FinBERT'] = result.label
    ##self.df_news['Fin_BERT_score'] = result.score
    return

  def BERT_quality_test(self, column = 'Title' ):
    from datasets import Dataset
    from transformers.pipelines.pt_utils import KeyDataset
    from transformers import pipeline

    nlp = pipeline("text-classification", model="ikoghoemmanuell/finetuned_fake_news_roberta",
                   device=-1, batch_size=1000)


    result = []

    dataset = Dataset.from_pandas(self.df_news)

    for out in tqdm(nlp(KeyDataset(dataset, 'Title'))):
      result.append(out)
    result = pd.DataFrame(result)

    self.df_news= self.df_news.join(result.set_axis(self.df_news.index))


    ##self.df_news['FakeBERT'] = result.label
    ##self.df_news['FakeBERT_score'] = result.score
    return

  def BERT_NER_test(self, column = 'Title' ):
    from datasets import Dataset
    from transformers.pipelines.pt_utils import KeyDataset
    from transformers import pipeline

    nlp = pipeline(task='ner',
               model='dslim/bert-base-NER', device=-1, batch_size=1000)

    result = []
    dataset = Dataset.from_pandas(self.df_news)

    for out in tqdm(nlp(KeyDataset(dataset, 'Title'))):
      out1 = [i['word'] for i in out]
      out2 = [i['entity'] for i in out]

      output = {}
      output =dict.fromkeys(out2, out1)
      result.append(output)

    result = pd.DataFrame(result)

    ##self.df_news = pd.concat([self.df_news, result], axis=1, ignore_index=True)

    self.df_news= self.df_news.join(result.set_axis(self.df_news.index))

    return

  def BERT_Relevance(self, column = 'Title' ):
    from datasets import Dataset
    from transformers.pipelines.pt_utils import KeyDataset
    from transformers import pipeline

    model_path = RELEVANCE_MODEL_PATH
    if not model_path.exists():
      raise FileNotFoundError(f"Relevance model directory not found at {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    device_id = 0 if torch.cuda.is_available() else -1
    nlp = pipeline("text-classification", model=model, tokenizer=tokenizer,
                  device=device_id, batch_size=1000)


    result = []

    ##some telegram news dont have title. pass source back
    ##self.df_news["Title"].fillna(self.df_news["Source"], inplace = True)
    self.df_news["Title"] = self.df_news["Title"] + self.df_news["Source"]

    dataset = Dataset.from_pandas(self.df_news)

    for out in tqdm(nlp(KeyDataset(dataset, 'Title'))):
      print(out)
      result.append(out)
    result = pd.DataFrame(result)

    self.df_news.drop(['label', "score"], axis=1, inplace=True)

    self.df_news= self.df_news.join(result.set_axis(self.df_news.index))


    ##self.df_news['FakeBERT'] = result.label
    ##self.df_news['FakeBERT_score'] = result.score
    return





  def save_BERT(self):
    output_path = BERT_RESULTS_PATH
    self.df_news.to_csv(output_path)
    mirror_to_drive(output_path)
    return

  def load_BERT(self):
    del self.df_news

    ##Currently we work with 100000 publications (less than 2% of dataset)
    ## It is done to ensure swift working of the functions bellow.
    self.df_news = pd.read_csv(BERT_RESULTS_PATH,
                               low_memory=False) ##, nrows=10000)

    self.df_news.set_index('Unnamed: 0',inplace=True)
    ##self.df_news.drop(['Unnamed: 0.1', 'index'], axis=1, inplace=True)




    ##self.df_news = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Data_paper/BERT_project_results.csv',
    ##                        low_memory=False)


    return

  def save_BERT_DBSCAN(self):
    output_path = BERT_DBSCAN_RESULTS_PATH
    self.df_news.to_csv(output_path)
    mirror_to_drive(output_path)
    return

  def load_BERT_DBSCAN(self):
    del self.df_news


    self.df_news = pd.read_csv(BERT_DBSCAN_RESULTS_PATH,
                               low_memory=False)
    self.df_news.set_index('Unnamed: 0',inplace=True)

    return






  ##Get Source feature, adjust time of publication
  def time_publication_adjustment(self):
    ## assume lag of google news is several hours
    self.df_news['Date'] = pd.to_datetime(self.df_news['Date'])


    ## move all publications of Saturday and Sunday to Monday
    print(self.df_news['Date']) ##+ pd.offsets.CustomBusinessDay(n=1, calendar=USFederalHolidayCalendar()))
    ##self.df_news.loc[:,'Date'] = self.df_news['Date']+ pd.offsets.BusinessDay(0)

    def offset_date(start, offset):
      return start + pd.offsets.CustomBusinessDay(n=offset, calendar=USFederalHolidayCalendar())

    offset = 0
    self.df_news['Date']= self.df_news['Date'].apply(lambda x: offset_date(x,
                                          offset))
    self.df_news['Date'] = self.df_news['Date'].dt.date
    print(self.df_news['Date'])

    self.df_news['Source'] = self.df_news['Title'].str.split(' - ').str[1]

    return



  ## request stock market data from
  def import_stock_data(self):
    Companies = self.df_news['Company'].unique()
    Companies = Companies.tolist()

    Stock_data = pd.DataFrame()

    ##Get index returns
    ticker = yf.Ticker('^GSPC')
    SP500 = ticker.history(period="5y")
    SP500["Returns"] = (SP500.Close - SP500.Open)/SP500.Open

    ##Technical transformations
    SP500 =SP500.reset_index()
    SP500.loc[:,'Date'] = pd.to_datetime(SP500.loc[:,'Date'])
    SP500['Date'] = SP500['Date'].dt.date


    ##stock data for each company
    for comp in Companies:
      try:
        ticker = yf.Ticker(comp)
        hist = ticker.history(period="5y")

        ##Technical transformations
        hist =hist.reset_index()
        hist.loc[:,'Date'] = pd.to_datetime(hist.loc[:,'Date'])
        ##print(hist['Date'])
        hist['Date'] = hist['Date'].dt.date
        hist['Date'] = pd.to_datetime(hist['Date'])

        ##print(hist)
        ##test_1 = apple.history(period="5y")
        ##test_1.set_index(pd.to_datetime(pd.to_datetime(test_1.reset_index()['Date']).dt.date), inplace=True)


        stock_count = pd.DataFrame(ticker.financials)
        ##print(stock_count)
        stock_count = stock_count.T
        stock_count = stock_count['Diluted Average Shares'].sort_index()
        stock_count = pd.DataFrame(stock_count)

        tol = pd.Timedelta('7 day')
        hist = pd.merge_asof(hist, stock_count, right_index=True,left_on='Date', direction='nearest',tolerance=tol)


        hist['Diluted Average Shares'].fillna(method='ffill', inplace=True)
        hist['Diluted Average Shares'].fillna(method='bfill', inplace=True)
        ##hist['Diluted Average Shares'].ffill(inplace=True).bfill(inplace=True)
        hist["Market_Cap"] = hist["Diluted Average Shares"] * hist["Close"]

        hist['Date'] = hist['Date'].dt.date


        try:
          Earn_releases = ticker.get_earnings_dates(limit=25)

          ##Technical transformations
          Earn_releases.reset_index(inplace=True)
          Earn_releases.rename(columns={'Earnings Date':'Date'}, inplace=True)
          Earn_releases.loc[:,'Date'] = pd.to_datetime(Earn_releases.loc[:,'Date'])
          Earn_releases['Date'] = Earn_releases['Date'].dt.date

          hist = pd.merge(hist, Earn_releases, how="left", on='Date')
        except:
          print("Error in parcing releases for", comp)

        ## append symbol
        hist['symbol'] = comp

        ##Add index returns for each company
        hist['SP500_returns'] = SP500["Returns"]
        hist['SP500_returns_yesterday'] = SP500["Returns"].shift(-1)

        ##Stock_data = Stock_data.append(hist) ##Change later
        ##print(hist)
        Stock_data = pd.concat([Stock_data, hist], ignore_index=True)
      except:
        print(comp, "not found on yfinance")

    ##Create returns variable
    Stock_data["Returns"] = (Stock_data.Close - Stock_data.Open)/Stock_data.Open
    Stock_data["Returns_binary"] = 0
    Stock_data.loc[Stock_data["Returns"]>0,["Returns_binary"]] = 1
    ## Create previos returns feature
    Stock_data["Returns_yesterday"] = Stock_data["Returns"].shift(-1)

    '''##Technical transformations
    Stock_data =Stock_data.reset_index()
    Stock_data.loc[:,'Date'] = pd.to_datetime(Stock_data.loc[:,'Date'])
    Stock_data['Date'] = Stock_data['Date'].dt.date'''

    self.df_stock = Stock_data
    return

  ##function to get features from BERT models, publishers
  def One_hot_encode(self, Interactions = True, OHE = True):
    ##self.df_news.Title.str.rsplit(" - ", n=1, expand=True)
    ##print(self.df_news.columns)
    self.df_news.reset_index(drop=True, inplace=True)
    for col in self.df_news.columns:
      print(col)

    ##We have too many Sources. We drop all except top 10.
    ##Credit to https://stackoverflow.com/questions/32511061/remove-low-frequency-values-from-pandas-dataframe

    '''threshold = 10 # Anything that occurs less than this will be removed.
    value_top10 = self.df_news.sort_values(['Source'], ascending=False).head(10)
    to_remove = self.df_news[self.df_news.Source != value_top10].index
    self.df_news.Source.replace(to_remove, 'Other', inplace=True)'''




    if OHE == True:

      ## init Encoder
      OHE = OneHotEncoder(categories='auto', sparse=True)

      ##selected columns for encoding
      columns_ohe= ['FinBERT', 'FakeBERT'] ## simple features currently.


      ## NOTE: OneHotEncoder has mamory problems on large dataset
      ## Current roundabout solution

      ##cycle to decrease memory usage.


      ##feature_arr = ohe.fit_transform(self.df_news[one_column]).toarray()
      OHE.fit(self.df_news[columns_ohe].head(1000))
      feature_arr = OHE.transform(self.df_news[columns_ohe]).toarray()
      feature_labels = OHE.categories_

      feature_labels = np.array(feature_labels, dtype=object).ravel()
      feature_labels = [item for sublist in feature_labels for item in sublist]
      ##print(feature_arr)
      ##print(feature_labels)
      features = pd.DataFrame(feature_arr, columns=feature_labels)
      print(features)

      ##self.df_news = pd.concat([self.df_news, features], axis=1)
      self.df_news = self.df_news.join(features.set_axis(self.df_news.index))

    else:

      cols1 = ['negative', 'neutral', 'positive']
      for col in tqdm(cols1):
        self.df_news[col] = 0
        self.df_news.loc[self.df_news['FinBERT'] == col, col] = 1

      cols2 = ['LABEL_0', 'LABEL_1']
      for col in tqdm(cols2):
        self.df_news[col] = 0
        self.df_news.loc[self.df_news['FakeBERT'] == col, col] = 1

      ## retired model
      '''cols3 = ['Rel_neg', 'Rel_neut', 'Rel_pos'] ## new names for Relevance model
      for index in tqdm([0, 1, 2]):
        New_name = cols3[index]
        Old_name = cols1[index]
        self.df_news[New_name] = 0
        self.df_news.loc[self.df_news['label'] == Old_name, New_name] = 1'''
      ##self.df_news[['negative', 'neutral', 'postive']]= pd.get_dummies(
       ##   self.df_news['FinBERT'], dtype=float)
      ##cols2 = ['LABEL_0', 'LABEL_1']
      ##self.df_news[['LABEL_0', 'LABEL_1']]= pd.get_dummies(
       ##   self.df_news['FakeBERT'], dtype=float)

    ##interactions between variables
    if Interactions == True:
      ## NER BERT: dummy = 1 if there is organisation in text
      self.df_news.loc[:, 'B-ORG'] = self.df_news['B-ORG'].notnull().astype('int')
      self.df_news.drop(['I-ORG', 'B-MISC', 'B-PER', 'I-PER',
                         'B-LOC', 'I-LOC', 'I-MISC'], axis=1,
                        inplace=True)

      self.df_news['positive*Label_1'] = self.df_news['positive'] * self.df_news['LABEL_1']
      self.df_news['negative*Label_1'] = self.df_news['negative'] * self.df_news['LABEL_1']
      return
    else:
      return

  ## merging dataframes to fit in classifiers
  def prepare_final(self, Interactions = True):

    ## To solve the error when running time_publication_adjustment(naive = False)
    self.df_news['Date'] = pd.to_datetime(self.df_news['Date']).dt.date


    ##Groupby and combine datasets
    New_indicators= self.df_news.groupby(['Company', 'Date']).sum()
    self.df = pd.merge(self.df_stock, New_indicators, left_on=['symbol', 'Date'],
                       right_on=['Company', 'Date'])
    self.df.set_index('Date', inplace = True)

    ## Market uncertanty index
    for date in self.df.index.unique():
      Set = self.df[self.df.index == date]
      Positive_sum = Set['positive'].sum()
      Negative_sum = Set['negative'].sum()
      try:
        Uncert = max(0, 1- Negative_sum / (Positive_sum + Negative_sum))
      except:
        Uncert = 0
      self.df.loc[self.df.index  == date, 'Uncert']  = Uncert

    if Interactions == True:
      self.df['positive*Label_1*Uncert'] = self.df['positive*Label_1'] * self.df.Uncert
      self.df['negative*Label_1*Uncert'] = self.df['negative*Label_1'] * self.df.Uncert
    else:
      pass


    ##drop outside columns: current stock data, leftover columns
    self.df.drop(['Open', 'High', 'Low', 'Close', 'Volume',
                  'Dividends', 'Stock Splits',
                  'Returns', 'symbol'], axis=1, inplace =True)
    try:
      self.df.drop([ 'Adj Close'], axis=1, inplace =True)
    except:
      pass
    return

  def train_test_split(self):

    X = self.df.drop(['Returns_binary'], axis=1).copy()
    y = self.df['Returns_binary'].copy()
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,
                                                                            y,
                                                                            test_size=0.2,
                                                                            random_state=42)

    return




  def LR_class(self, naive=True, metric = "f1_weighted"):

    ## Running StandardScaler() for Loogistic Regression
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(self.X_train)
    X_scaled = scaler.transform(self.X_train)
    self.X_train = X_scaled

    ##run basic model
    if naive == True:
      LR_final = LogisticRegression()
      scores = cross_val_score(LR_final, self.X_train, self.y_train,
                              cv=5 , scoring=metric)
      self.LR_cv_results = scores
      print("Results of basic Logistic Regression:")
      print('Metric: ', metric, " mean score of cross val: ", scores.mean())

    else:
        ##initiation of optuna for hyperparameter tunning

        ##objective function for maximisation
        def objective(trial):

          X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train,
                                                        test_size=0.2,
                                                        random_state=42)

          params = {
              'tol' : trial.suggest_float('tol' , 1e-6 , 1e-3),
              'C' : trial.suggest_float("C", 1e-2, 1, log=True)}
          model = LogisticRegression(**params)
          model.fit(X_train, y_train)
          y_pred = model.predict(X_val)


          f1_result = f1_score(y_val, y_pred, average='weighted')

          return f1_result   ##f1_score(self.y_train, y_pred)

        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=100)

        ##optuna results
        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
          print("    {}: {}".format(key, value))

        ##feed best parameters for cross validation
        params_final = study.best_trial.params
        LR_final = LogisticRegression(**params_final)


        scores = cross_val_score(LR_final, self.X_train, self.y_train,
                                cv=5 , scoring="f1_weighted")

        self.LR_cv_results = scores

        LR_final.fit(self.X_train, self.y_train)
        ##Report
        print("Results of hyper-tuned Logistic Regression:")

        ## Train results
        print('Metric: ', metric, " mean score of cross val: ", scores.mean())


        roc_train = roc_auc_score(self.y_train,
                                LR_final.predict_proba(self.X_train)[:, 1])
        Gini_train = 2*roc_train -1

        print('ROC_AUC train: ',roc_train)
        print('Gini train: ',Gini_train)
        print('F1 train (cross validation): ',scores.mean())

        ## Test results
        roc_test =roc_auc_score(self.y_test,
                                  LR_final.predict_proba(scaler.transform(self.X_test))[:, 1])
        Gini_test = 2*roc_test -1

        y_pred = LR_final.predict(scaler.transform(self.X_test))
        f1_result = f1_score(self.y_test, y_pred, average='weighted')

        print('')
        print('Results on the holdout sample')
        print('ROC_AUC test: ',roc_test)
        print('Gini test: ',Gini_test)
        print('F1-score test: ',f1_result)
        return




  def RF_class(self, naive=True, metric = "f1_weighted"):

    ##run basic model
    if naive == True:
      LR_final = RandomForestClassifier(random_state=42)
      scores = cross_val_score(LR_final, self.X_train, self.y_train,
                              cv=5 , scoring=metric)
      self.LR_cv_results = scores
      print("Results of basic Random Forest:")
      print('Metric: ', metric, " mean score of cross val: ", scores.mean())

      print(LR_final.predict_proba(self.X_train)[:, 1])
      roc_train = roc_auc_score(self.y_train,
                                LR_final.predict_proba(self.X_train)[:, 1])
      Gini_train = 2*roc_train -1

      print('ROC_AUC train: ',roc_train)
      print('Gini train: ',Gini_train)

      roc_test =roc_auc_score(self.y_test,
                                LR_final.predict_proba(scaler.transform(self.X_test))[:, 1])
      Gini_test = 2*roc_test -1

      print('ROC_AUC test: ',roc_test)
      print('Gini test: ',Gini_test)
      return
    else:
      ##initiation of optuna for hyperparameter tunning

      ##objective function for maximisation
      def objective(trial):
        ##test validation split for optuna
        X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train,
                                                        test_size=0.2,
                                                        random_state=42)

        params = {
            'n_estimators' : trial.suggest_int('n_estimators' , 5, 200),
            'min_samples_split': trial.suggest_int('min_samples_split', 3, 30),
            'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 3, 10),
            'max_depth' : trial.suggest_int('max_depth', 3, 30),
            'bootstrap' : True,
            'random_state' : 42
        }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        f1_result = f1_score(y_val, y_pred, average='weighted')
        return f1_result

      sampler = TPESampler(seed=123)
      study = optuna.create_study(direction='maximize', sampler=sampler)
      study.optimize(objective, n_trials=100)

      ##optuna results
      print("Number of finished trials: {}".format(len(study.trials)))

      print("Best trial:")
      trial = study.best_trial

      print("  Value: {}".format(trial.value))

      print("  Params: ")
      for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
      params_final = study.best_trial.params

      ##feed best parameters for cross validation
      RF_final = RandomForestClassifier(**params_final)
      RF_final = RandomForestClassifier(**params_final)


      scores = cross_val_score(RF_final, self.X_train, self.y_train,
                              cv=5, scoring="f1_weighted")

      RF_final.fit(self.X_train, self.y_train)
      ##report


      self.RF_cv_results = scores
      print("Results of hyper-tuned Random Forest:")
      print('Metric: ', metric, " mean score of cross val: ", scores.mean())
      ## Train results
      roc_train = roc_auc_score(self.y_train,
                                RF_final.predict_proba(self.X_train)[:, 1])
      Gini_train = 2*roc_train -1

      print('ROC_AUC train: ',roc_train)
      print('Gini train: ',Gini_train)

      ## Test results
      roc_test =roc_auc_score(self.y_test,
                                RF_final.predict_proba(self.X_test)[:, 1])
      Gini_test = 2*roc_test -1
      y_pred = RF_final.predict(self.X_test)
      f1_result = f1_score(self.y_test, y_pred, average='weighted')

      print('')
      print('Results on the holdout sample')
      print('ROC_AUC test: ',roc_test)
      print('Gini test: ',Gini_test)
      print('F1-score test: ',f1_result)


      ##Plot feature importance
      feature_names = self.X_train.columns
      importances = RF_final.feature_importances_
      forest_importances = pd.Series(importances, index=feature_names)

      import matplotlib.pyplot as plt
      fig, ax = plt.subplots()
      std = np.std([importances for tree in RF_final.estimators_], axis=0)
      forest_importances.plot.bar(yerr=std, ax=ax)
      ax.set_title("Feature importances for RF")
      ax.set_ylabel("Mean decrease in impurity")
      fig.tight_layout()
      plt.show()
      return

  def LightGBM(self):

    ## init dataset for LightGBM
    dtrain = lgb.Dataset(self.X_train, label=self.y_train)

    def objective(trial):


      ##test validation split for optuna
      X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train,
                                                        test_size=0.2,
                                                        random_state=42)

      dtrain = lgb.Dataset(X_train, label=y_train)

      ## params to optimize
      param = {
          "objective": "binary",
          "metric": "binary_logloss",
          "verbosity": -1,
          "boosting_type": "gbdt",
          "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
          "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
          "num_leaves": trial.suggest_int("num_leaves", 2, 256),
          "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
          "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
          "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
          "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
          'verbose': -1
          }
      ##fit LightGBM in trial
      gbm = lgb.train(param, dtrain)
      preds = gbm.predict(X_val)
      pred_labels = np.rint(preds)
      f1_result = f1_score(y_val, pred_labels, average='weighted')
      return f1_result



    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)

    ##optuna results
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    self.best_LightGBM_params = trial.params


    ## CV results
    cv_results = lgb.cv(
        trial.params,
        dtrain,
        num_boost_round=100,
        nfold=5,
        metrics='f1')
    print(cv_results)
    gbm = lgb.train(trial.params, dtrain)
    pred = gbm.predict(self.X_train)

    ##Binary prediction
    pred_binary = [1 if i>0.5 else 0 for i in pred]


    f1_result =f1_score(self.y_train, pred_binary, average='weighted')


    print("Results of hyper-tuned LightGBM:")

    ## Train results
    roc_train = roc_auc_score(self.y_train, pred)
    Gini_train = 2*roc_train -1

    print('ROC_AUC train: ',roc_train)
    print('Gini train: ',Gini_train)
    print('F1-score train', f1_result)

    ## Test results
    pred = gbm.predict(self.X_test)

    ##Binary prediction
    pred_binary = [1 if i>0.5 else 0 for i in pred]

    roc_test =roc_auc_score(self.y_test, pred_binary)
    Gini_test = 2*roc_test -1

    f1_result = f1_score(self.y_test, pred_binary, average='weighted')

    print('')
    print('Results on the holdout sample')
    print('ROC_AUC test: ',roc_test)
    print('Gini test: ',Gini_test)
    print('F1-score test: ',f1_result)


    return

ML = ML_build()
ML.news_df(News_dataset.head(5)) ##init with small size
ML.load_BERT()
if USE_PRECOMPUTED_OHE and BERT_OHE_PATH.exists():
  precomputed_path = BERT_OHE_PATH
  precomputed_df = pd.read_csv(precomputed_path, low_memory=False)
  if 'Unnamed: 0' in precomputed_df.columns:
    precomputed_df.drop(columns=['Unnamed: 0'], inplace=True)
  ML.df_news = precomputed_df.copy()
  if 'Date' in ML.df_news.columns:
    ML.df_news['Date'] = pd.to_datetime(ML.df_news['Date'], errors='coerce')
  ML.df_news.reset_index(drop=True, inplace=True)
else:
  ML.One_hot_encode(OHE=False)
  ML.df_news.to_csv(BERT_OHE_PATH, index=False)

##Import stock data and merge data on Ticker + Date
ML.import_stock_data()

ML.df_stock

from inspect import EndOfBlock
import scipy.stats as sci

df_stock = ML.df_stock.copy()
##df_stock = df_stock.drop('Adj Close', axis=1)
##df_stock.dropna()

def Beta_calculation(df,window, stock='Returns', market ='SP500_returns'):


  symbols = df['symbol'].dropna().unique()
  df.set_index(['symbol', "Date"],  inplace=True)



  df['Beta'] = np.nan
  df['AR'] = np.nan
  df['AR_signif'] = np.nan

  est_window = 50
  event_window = 10

  df['Event'] = np.nan
  df['Event_ID'] = np.nan





  test_df = df.copy()
  risk_free = yf.Ticker('^IRX')
  risk_free = risk_free.history(period="5y")['Close'] *0.01
  risk_free =risk_free.reset_index()
  risk_free.loc[:,'Date'] = pd.to_datetime(risk_free.loc[:,'Date'])
  risk_free['Date'] = risk_free['Date'].dt.date

  unique_ID = 0
  for i in symbols:

    unique_ID = round(unique_ID/100)*100
    unique_ID += 100


    print(i)
    sliced_by_company = df.loc[i].copy()
    sliced_by_company.reset_index(inplace=True)



    print(sliced_by_company.index[sliced_by_company['Reported EPS'].notna()].tolist())
    ##print(sliced_by_company[sliced_by_company['Reported EPS'].notna()])

    for event in sliced_by_company.index[sliced_by_company['Surprise(%)'].notna()].tolist():

      unique_ID += 1

      Start_of_slice = int(event- event_window/2 -est_window)
      End_of_slice = int(event+ event_window/2)

      ##print(Start_of_slice,int(event-event_window/2),  End_of_slice)
      sliced = sliced_by_company[Start_of_slice : End_of_slice].copy()

      sliced["Event_ID"] = unique_ID
      sliced.loc[int(event-event_window/2)  : End_of_slice, "Event" ] = 1

      stock_var = sliced.loc[Start_of_slice: int(event-event_window/2), stock]
      market_var = sliced.loc[Start_of_slice: int(event-event_window/2), market]

      '''covariance = sliced[stock].cov(sliced[market])
      variance = sliced[market].var()'''

      covariance = stock_var.cov(market_var)
      variance = market_var.var()

      ##print(covariance, variance)

      beta = covariance/variance
      sliced['Beta']= beta
      if i == 'AAPL':
        print(beta)
        ##print(sliced)
      ##df.loc[df['symbol'] == i, 'Beta'] = beta



      ##print(sliced)
      ##print(risk_free[risk_free['Date'].isin(sliced['Date'])])
      '''sliced['Risk_free']= risk_free[risk_free['Date'].isin(sliced['Date'])]['Close'].copy()
      try:
        sliced['Risk_free']= risk_free[risk_free['Date'].isin(sliced['Date'])]['Close'].to_list()
      except:
        ##print(sliced['Risk_free'])
        print(len(sliced['Date']))
        sliced = sliced[sliced['Date'].isin(risk_free['Date'])]
        print(len(sliced['Date']))

        sliced['Risk_free']= risk_free[risk_free['Date'].isin(sliced['Date'])]['Close'].to_list()'''

      ## CAPM with risk-free
      ##sliced['model_ret'] = sliced['Risk_free'] + sliced['Beta'] *( sliced[market] - sliced['Risk_free'])

      sliced['model_ret'] = 0 + sliced['Beta'] *( sliced[market] - 0)
      sliced['AR'] = sliced['Returns'] - sliced['model_ret']
      sliced['AR'] = sliced['AR']*100



      crit_val = sci.norm.ppf(0.95)
      def test_crit(stat, crit):
        if stat >= crit:
          return 1
        elif stat <= -crit:
          return 2
        else:
          return 0
      sliced['AR_signif'] = sliced['AR'].apply(lambda x: test_crit(x, crit_val))

      sliced_by_company.update(sliced, overwrite=False)


      if i == 'AAPL':
        test_df = df.copy()

      print(unique_ID)

    sliced_by_company['symbol'] = i
    sliced_by_company.set_index(['symbol', 'Date'], inplace=True)

    try:
      df.update(sliced_by_company, overwrite=False)
    except:
      print(sliced_by_company)


  df.reset_index(inplace= True)
  return df, test_df
df_stock, test_df = Beta_calculation(df_stock, 30)

df_stock.to_csv(DF_STOCK_PATH, index=True)
df_stock = pd.read_csv(DF_STOCK_PATH)
df_stock['Date'] = pd.to_datetime(df_stock.loc[:,'Date'])
df_stock['Date'] = df_stock['Date'] .dt.date

def prepare_news_df( curated= True, Organisation = True):
  News_df = ML.df_news.copy()
  News_df =News_df.reset_index()

  Date_col= pd.to_datetime(News_df['Date'], utc=True, errors='coerce')
  News_df['Date'] = Date_col.dt.date

  ## Copy the colummns
  News_df['Rel_pos'] = News_df['positive']
  News_df['Rel_neg'] = News_df['negative']
  ##News_df['Rel_pos'] = News_df['negative']





  News_df['positive*Label_0'] = News_df['positive'] * News_df['LABEL_0']
  News_df['negative*Label_0'] = News_df['negative'] * News_df['LABEL_0']


  ##Relevance adjustment
  News_df['Rel_pos*Label_1'] = News_df['Rel_pos'] * News_df['LABEL_1']
  News_df['Rel_neg*Label_1'] = News_df['Rel_neg'] * News_df['LABEL_1']
  News_df['Rel_pos*Label_0'] = News_df['Rel_pos'] * News_df['LABEL_0']
  News_df['Rel_neg*Label_0'] = News_df['Rel_neg'] * News_df['LABEL_0']

  Original_News_df = News_df.copy()

  if Organisation == True:
    News_df = News_df[News_df['B-ORG'] == 1]

  if curated==True:
    if not SOURCES_CLASS_PATH.exists():
      raise FileNotFoundError(f"Sources_class.xlsx not found at {SOURCES_CLASS_PATH}")
    sources_df = pd.read_excel(SOURCES_CLASS_PATH, header=1).fillna(0)

    News_df = News_df.merge(sources_df, on="Source", how='left')
    News_df =News_df.drop("Source", axis=1)
    News_df =News_df.drop("count", axis=1)
    News_df = News_df[(News_df['Relevant'] >0) | (News_df['Traditional'] >0)]

  News_df = News_df[["Date", "Company", 'positive', 'negative',
                    'positive*Label_1', 'negative*Label_1', 'positive*Label_0','negative*Label_0',
                    'Rel_pos*Label_1', 'Rel_neg*Label_1', 'Rel_pos*Label_0', 'Rel_neg*Label_0'
                    ]]


  News_df= News_df.groupby(['Company', 'Date']).sum()

  return News_df, Original_News_df

def merge_stock_news(news_df, stock_df, only_rel =True, no_Events=False ):

  df_stock = pd.merge(stock_df, news_df.reset_index(), left_on=['symbol', 'Date'],
                       right_on=['Company', 'Date'], how="left")


  print(df_stock)
  df_stock = df_stock[df_stock['Date'] >= pd.to_datetime('2020-01-01').date() ]
  df_stock = df_stock[df_stock['Date'] <= pd.to_datetime('2023-09-01').date() ]


  ## Market uncertanty index

  Uncertainty_df = pd.DataFrame()
  for date in df_stock.Date.unique():
    Set = df_stock[df_stock.Date == date]
    Positive_sum = Set['positive'].sum()
    Negative_sum = Set['negative'].sum()
    try:

      Uncert = max(0, 1- Negative_sum / (Positive_sum + Negative_sum))
      ##print(Positive_sum, Negative_sum, Uncert)

    except:
      Uncert = 0
    df_stock.loc[df_stock.Date == date, 'Uncert']  = Uncert
    day_dict = {"Date": date, "Uncert": Uncert}
    Uncertainty_df = pd.concat([Uncertainty_df, pd.DataFrame([day_dict])], ignore_index=True)
    ##Uncertainty_df = Uncertainty_df.append(day_dict)

  print(Uncertainty_df)

  ## Event window

  try:
    df_stock["Event_only_index"] =np.nan
    df_stock.loc[(df_stock["Event"]==1) & (df_stock["Event_ID"].notna()), "Event_only_index"] = df_stock["Event_ID"]

  except:
    print('Error in EVENT_ID column')


  ## Amihud computation

  df_stock["Illiquidity"] = abs(df_stock["Returns"])/(df_stock["High"] * df_stock["High"]*df_stock["Volume"]/2)

  grouped = df_stock.drop("Date", axis=1).groupby('symbol')
  print(grouped)


  results = {'symbol':[], 'illiquidity':[]}
  for item, grp in grouped:
    print(item)
    ##print(grp.tail(2))

    ##subset_mean = grp.tail(2).sum()[0]
    subset_mean =grp["Illiquidity"].mean()
    print(subset_mean)
    results['symbol'].append(item)
    results['illiquidity'].append(subset_mean)

  print(results)
  res_df = pd.DataFrame(results)
  Quant = res_df["illiquidity"].quantile(0.75)
  res_df['Illiq'] = res_df['illiquidity'].ge(Quant).astype(int)
  df_stock = df_stock.merge(res_df, on='symbol', how='left')


  ##Sentiment index
  New_cols = ["PL1_", "NL1_", "PL0_", "NL0_", "RPL1_","RNL1_", "RPL0_", "RNL0_"]
  old_cols = ['positive*Label_1', 'negative*Label_1', 'positive*Label_0', 'negative*Label_0',
              'Rel_pos*Label_1', 'Rel_neg*Label_1','Rel_pos*Label_0', 'Rel_neg*Label_0'
              ]
  for lag in [20, 40, 60, 120]:
    names = [ i + str(lag) for i in New_cols]
    print(names)
    df_stock[names] = df_stock[old_cols].rolling(window=lag).sum()

  df_stock.reset_index(inplace=True)



  if no_Events == True:

    df_stock["Date"] = pd.to_datetime(df_stock["Date"])

    test = df_stock[["Date",  'Volume', 'Dividends', 'Stock Splits', 'Market_Cap', 'SP500_returns', 'SP500_returns_yesterday', 'Returns', 'AR', 'EPS Estimate', 'Reported EPS',
       'Surprise(%)', 'symbol', 'positive*Label_1',
       'negative*Label_1', 'positive*Label_0', 'negative*Label_0', 'Uncert',
       'Event_only_index', 'Rel_pos*Label_1', 'Rel_neg*Label_1',
       'Rel_pos*Label_0', 'Rel_neg*Label_0', "Illiq",'PL1_20', 'NL1_20', 'PL0_20',
       'NL0_20', 'RPL1_20', 'RNL1_20', 'RPL0_20', 'RNL0_20', 'PL1_40',
       'NL1_40', 'PL0_40', 'NL0_40', 'RPL1_40', 'RNL1_40', 'RPL0_40',
       'RNL0_40', 'PL1_60', 'NL1_60', 'PL0_60', 'NL0_60', 'RPL1_60', 'RNL1_60',
       'RPL0_60', 'RNL0_60', 'PL1_120', 'NL1_120', 'PL0_120', 'NL0_120',
       'RPL1_120', 'RNL1_120', 'RPL0_120', 'RNL0_120']]
    model_run = test.groupby(['symbol', 'Date']).sum()
    ##return model_run, df_stock

  else:

    test = df_stock[['Date', 'Volume', 'Dividends', 'Stock Splits', 'Market_Cap', 'SP500_returns', 'SP500_returns_yesterday', 'Returns', 'AR', 'EPS Estimate', 'Reported EPS',
        'Surprise(%)', 'symbol', 'Event', 'Event_ID', 'positive*Label_1',
        'negative*Label_1', 'positive*Label_0', 'negative*Label_0', 'Uncert',
        'Event_only_index', 'Rel_pos*Label_1', 'Rel_neg*Label_1',
        'Rel_pos*Label_0', 'Rel_neg*Label_0', "Illiq",'PL1_20', 'NL1_20', 'PL0_20',
        'NL0_20', 'RPL1_20', 'RNL1_20', 'RPL0_20', 'RNL0_20', 'PL1_40',
        'NL1_40', 'PL0_40', 'NL0_40', 'RPL1_40', 'RNL1_40', 'RPL0_40',
        'RNL0_40', 'PL1_60', 'NL1_60', 'PL0_60', 'NL0_60', 'RPL1_60', 'RNL1_60',
        'RPL0_60', 'RNL0_60', 'PL1_120', 'NL1_120', 'PL0_120', 'NL0_120',
        'RPL1_120', 'RNL1_120', 'RPL0_120', 'RNL0_120']]

    test2 = test.drop(test[test['Event_only_index'] == 0.0].index)
    ##model_run = test2.groupby(['symbol', 'Event_only_index']).sum()

    ##text_cols = df.columns[1:3]
    data_cols = set(test2.columns) - set(["Date", 'symbol'])
    d1 = dict.fromkeys(data_cols, 'sum')
    d2 = dict.fromkeys(["Date"], 'first')
    d = {**d1, **d2}

    model_run = test2.groupby(['symbol', 'Event_only_index']).agg(d)
    print(model_run)

    ## Fill in uncertainty index
    model_run.drop("Uncert", axis=1, inplace=True)

    def get_uncertainty(df, df_uncert):

      date = df['Date']
      Index_value = df_uncert[df_uncert['Date'] == date]['Uncert'].values[0]
      print(date, Index_value)
      return Index_value

    model_run['Uncert']= model_run.apply(lambda x: get_uncertainty(x, Uncertainty_df), axis =1)
  ##model_run= model_run.merge(Uncertainty_df, how='left', on=['Date'])
  print(model_run)
  ##model_run.set_index(['symbol', 'Event_only_index'], inplace=True)




  ##functions for other columns
  def new_dummy(df, key_col, col_to_apply):
    for i in col_to_apply:
      name = i +"_" + key_col
      print(name)
      df[name] = df[key_col]* df[i]
    return df


  def new_index(df, Positive, all_cols, scale=10):
    for i in Positive:
      name = i +"_" + "index"
      print(name)

      Index_of_Negative = all_cols.index(i) +1
      neg_col = all_cols[Index_of_Negative]
      df[name] = (df[i] - df[neg_col])/ (df[i] + df[neg_col])/10
    return df

  Lag_list = ['PL1_20', 'NL1_20',
       'PL0_20', 'NL0_20', 'RPL1_20', 'RNL1_20', 'RPL0_20', 'RNL0_20',
       'PL1_40', 'NL1_40', 'PL0_40', 'NL0_40', 'RPL1_40', 'RNL1_40', 'RPL0_40',
       'RNL0_40', 'PL1_60', 'NL1_60', 'PL0_60', 'NL0_60', 'RPL1_60', 'RNL1_60',
       'RPL0_60', 'RNL0_60', 'PL1_120', 'NL1_120', 'PL0_120', 'NL0_120',
       'RPL1_120', 'RNL1_120', 'RPL0_120', 'RNL0_120']

  Positive_values = ['PL1_20', 'PL0_20',  'RPL1_20',  'RPL0_20',
        'PL1_40', 'PL0_40',  'RPL1_40', 'RPL0_40', 'PL1_60', 'PL0_60', 'RPL1_60',
        'RPL0_60', 'PL1_120', 'PL0_120', 'RPL1_120',  'RPL0_120']
  Negative_values = list(set(Lag_list) - set(Positive_values))
  model_run = new_index(model_run, Positive_values, Lag_list)



  index_list =['PL1_20_index', 'PL0_20_index','RPL1_20_index', 'RPL0_20_index',
               'PL1_40_index', 'PL0_40_index', 'RPL1_40_index', 'RPL0_40_index',
               'PL1_60_index', 'PL0_60_index', 'RPL1_60_index', 'RPL0_60_index',
               'PL1_120_index', 'PL0_120_index','RPL1_120_index', 'RPL0_120_index']

  model_run = new_dummy(model_run, "Uncert", index_list)
  model_run = new_dummy(model_run, "Illiq", index_list)


  Col_list = ['positive*Label_1', 'negative*Label_1', 'positive*Label_0',
              'negative*Label_0', 'Rel_pos*Label_1', 'Rel_neg*Label_1',
              'Rel_pos*Label_0', 'Rel_neg*Label_0']
  model_run = new_dummy(model_run, "Uncert", Col_list)
  model_run = new_dummy(model_run, "Illiq", Col_list)




  return model_run, df_stock

News_df = prepare_news_df(curated= True, Organisation = True)

model_run = merge_stock_news(News_df[0], df_stock, only_rel=False, no_Events=False )

try:
  import linearmodels as lm
except ImportError as exc:
  raise ImportError("linearmodels is required. Please install it via 'pip install linearmodels'.") from exc
model_run[0]['const'] = 1
model_run[0]['Market_Cap'] = np.log(model_run[0]['Market_Cap'])
model_run[0]['Volume'] = np.log(model_run[0]['Volume'])

model_run_final = model_run[0].copy()
model_run_final.columns = model_run[0].columns.str.lower().str.replace('*','_').str.replace('(%)','_percent')

m = lm.PanelOLS.from_formula('ar ~ const +   dividends + volume + market_cap+ surprise_percent + positive_label_1 + negative_label_1 + EntityEffects', data = model_run_final, check_rank=False, drop_absorbed=True)
H100  = m.fit(cov_type='clustered', cluster_entity=True )
print(H100)

m = lm.PanelOLS.from_formula('ar ~ const +   dividends + volume + market_cap+ surprise_percent + positive_label_1 + negative_label_1 + positive_label_0  + EntityEffects', data = model_run_final, check_rank=False, drop_absorbed=True)
H101  = m.fit(cov_type='clustered', cluster_entity=True )
print(H101)

m = lm.PanelOLS.from_formula('ar ~ const +   dividends + volume + market_cap+ surprise_percent + positive_label_1 + negative_label_1 + negative_label_0 + EntityEffects', data = model_run_final, check_rank=False, drop_absorbed=True)
H102  = m.fit(cov_type='clustered', cluster_entity=True )
print(H102)

m = lm.PanelOLS.from_formula('ar ~ const +   dividends  + market_cap+ volume + surprise_percent + positive_label_1 + negative_label_1 + positive_label_0 + negative_label_0 + EntityEffects', data = model_run_final, check_rank=False, drop_absorbed=True)
H103  = m.fit(cov_type='clustered', cluster_entity=True )
print(H103)



m = lm.PanelOLS.from_formula('ar ~ const +   dividends + volume + market_cap+ surprise_percent + rel_pos_label_1 + rel_neg_label_1  + EntityEffects', data = model_run_final, check_rank=False, drop_absorbed=True)
H104  = m.fit(cov_type='clustered', cluster_entity=True )
print(H104)


m = lm.PanelOLS.from_formula('ar ~ const +   dividends + volume + market_cap+ surprise_percent+ rel_pos_label_1 + rel_neg_label_1 + rel_pos_label_0 + EntityEffects', data = model_run_final, check_rank=False, drop_absorbed=True)
H105  = m.fit(cov_type='clustered', cluster_entity=True )
print(H105)


m = lm.PanelOLS.from_formula('ar ~ const +   dividends + volume + market_cap+ surprise_percent+ rel_pos_label_1 + rel_neg_label_1 + rel_neg_label_0 + EntityEffects', data = model_run_final, check_rank=False, drop_absorbed=True)
H106  = m.fit(cov_type='clustered', cluster_entity=True )
print(H106)

m = lm.PanelOLS.from_formula('ar ~ const +   dividends + volume + market_cap+ surprise_percent + rel_pos_label_1 + rel_neg_label_1 + rel_pos_label_0 + rel_neg_label_0 + EntityEffects', data = model_run_final, check_rank=False, drop_absorbed=True)
H107  = m.fit(cov_type='clustered', cluster_entity=True )
print(H107)



report = Stargazer([H100, H101, H102, H103, H104, H105, H106, H107])
##report.render_html()
with open("Report0.html", "w") as file:
    file.write(report.render_html())
