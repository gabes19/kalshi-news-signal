from transformers import pipeline
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import requests

#headlines extractions and sentiment classification

load_dotenv()
gnews_api_key = os.getenv("GNEWS_API_KEY")
pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")

#Keywords related to inflation used to identify relevant articles
keywords = ["cpi", "inflation", "consumer price", "pce", "price index", "ppi", "rate hike", "monetary policy",
            "fed", "federal reserve", "interest rate", "deflation"]

#Headline LEVEL DATA
#Headline level: Date published, id, title, description, sentiment, source name, title + description sentiment


def get_headlines(keywords:list[str],from_date:str,to_date:str)->list[str]:
  keywords_string = " OR ".join(keywords)
  params = {"in": "title,description", "from": from_date, "to": to_date, "lang" : "en", "q":keywords_string,
              "country":"us", "max":100, "apikey":gnews_api_key}
  response = requests.get("https://gnews.io/api/v4/search", params=params)
  data = response.json()
  headlines = {}
  for article in data['articles']:
      headlines['date'] = article['publishedAt']
      headlines['id'] = article['id']
      headlines['title'] = article['title']
      headlines['description'] = article['description']
      headlines['source'] = article['source']['name']
  return headlines

def get_sentiment(text:str):
   map = {"Very Negative": 0, "Negative":0.25, "Neutral":0.5, "Positive":0.75, "Very Positive": 1}
   result = pipe(text)[0]['label']
   sentiment_number = map.get(result)
   return sentiment_number


# print(get_headlines(keywords=keywords,from_date="2025-10-01T00:00:00.000Z",to_date="2025-11-05T00:00:00.000Z"))
print(get_sentiment('Fed’s Paulson Backs Two More 2025 Rate Cuts Despite Tariffs Federal Reserve Bank of Philadelphia President Anna Paulson signaled she favors two more quarter-point interest-rate cuts this year, as monetary policy should look through the impact of tariffs in consumer price increases.'))




