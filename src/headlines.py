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

#Takes in keyword list, and dates and outputs a list of dicts containing headline info
def get_headlines(keywords:list[str],from_date:str,to_date:str)->list[dict]:
  keywords_string = " OR ".join(keywords)
  params = {"in": "title,description", "from": from_date, "to": to_date, "lang" : "en", "q":keywords_string,
              "country":"us", "max":100, "apikey":gnews_api_key}
  response = requests.get("https://gnews.io/api/v4/search", params=params)
  data = response.json()
  headlines = []
  for article in data['articles']:
      headline = {}
      headline['date'] = article['publishedAt']
      headline['id'] = article['id']
      headline['title'] = article['title']
      headline['description'] = article['description']
      headline['source'] = article['source']['name']
      headlines.append(headline)
  return headlines

def get_sentiment(text:str)->int:
  map = {"Very Negative": 0, "Negative":0.25, "Neutral":0.5, "Positive":0.75, "Very Positive": 1}
  result = pipe(text)[0]['label']
  sentiment_number = map.get(result)
  return sentiment_number

def collect_all_headline_data(keywords:list[str],from_date:str,to_date:str):
  headlines = get_headlines(keywords=keywords, from_date=from_date,to_date=to_date)
  for headline in headlines:
    text = headline['title'] + " " + headline['description']
    sentiment = get_sentiment(text)
    headline['sentiment'] = sentiment
  df = pd.DataFrame(headlines)
  return df


# print(get_headlines(keywords=keywords,from_date="2025-10-01T00:00:00.000Z",to_date="2025-11-05T00:00:00.000Z"))
# print(get_sentiment('Fed’s Paulson Backs Two More 2025 Rate Cuts Despite Tariffs Federal Reserve Bank of Philadelphia President Anna Paulson signaled she favors two more quarter-point interest-rate cuts this year, as monetary policy should look through the impact of tariffs in consumer price increases.'))
print(collect_all_headline_data(keywords=keywords,from_date="2025-10-01T00:00:00.000Z",to_date="2025-11-05T00:00:00.000Z"))




