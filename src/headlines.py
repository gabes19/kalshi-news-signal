from transformers import pipeline
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from newsapi import NewsApiClient

#headlines extractions and sentiment classification

load_dotenv()
news_api_key = os.getenv("NEWS_API_KEY")
api = NewsApiClient(news_api_key)
pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")

#Keywords related to inflation used to identify relevant articles
keywords = ["cpi", "inflation", "consumer price", "pce", "price index", "ppi",
            "cost of living", "price pressure", "rate hike", "monetary policy",
            "fed", "federal reserve", "interest rate", "disinflation", "deflation"]

#Source IDs of top US news sources by NewsApi (used to filter relevant US inflation news)
source_ids = ['abc-news', 'al-jazeera-english', 'ars-technica', 'associated-press',
               'axios', 'bleacher-report', 'bloomberg', 'breitbart-news', 
               'business-insider', 'buzzfeed', 'cbs-news', 'cnn', 'crypto-coins-news', 
               'engadget', 'entertainment-weekly', 'espn', 'espn-cric-info', 'fortune',
                 'fox-news', 'fox-sports', 'google-news', 'hacker-news', 'ign', 
                 'mashable', 'medical-news-today', 'msnbc', 'mtv-news', 
                 'national-geographic', 'national-review', 'nbc-news', 
                 'new-scientist', 'newsweek', 'new-york-magazine', 
                 'next-big-future', 'nfl-news', 'nhl-news', 'politico',
                   'polygon', 'recode', 'reddit-r-all', 'reuters', 'techcrunch', 
                   'techradar', 'the-american-conservative', 'the-hill',
                 'the-huffington-post', 'the-next-web', 'the-verge',
                   'the-wall-street-journal', 'the-washington-post',
                     'the-washington-times', 'time', 'usa-today', 'vice-news', 'wired']

articles = []

#TODO
def get_headlines(sources:list[str],keywords:list[str],day:str)->list[str]:
    ''' Gets all headlines from specified sources on a given day containing keywords in the title
    Args: keywords - list of string keywords
          sources: list of string source ids from NewsApi
          day - string written in "YYYY-MM-DD" format

    Returns: a list of string headlines on that day
    '''
for word in keywords:
    response = api.get_everything(qintitle=word, language= "en", from_param = "2025-11-04", to = "2025-11-05")
    articles.append(response)
if articles:
    print(articles[0])
else:
    print("None..")

