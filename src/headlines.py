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
  try:
    keywords_string = " OR ".join(keywords) if keywords else 'news'

    headlines = []
    page = 1
    while True:
      params = {"in": "title,description", "from": from_date, "to": to_date, "lang" : "en", "q":keywords_string,
                "country":"us", "max":100, "apikey":gnews_api_key, "page":page} 
      data = requests.get("https://gnews.io/api/v4/search", params=params).json()
      articles = data.get("articles", [])
      if not articles:
          break
      for a in articles:
          headline = {}
          headline['date'] = a['publishedAt']
          headline['id'] = a['id']
          headline['title'] = a['title']
          headline['description'] = a['description']
          headline['source'] = a['source']['name']
          headlines.append(headline)
      page += 1
      if page > 10:
        break

    return headlines
  except Exception as e:
    print(f'{e}')

def get_sentiment(text:str)->int:
  map = {"Very Negative": 0, "Negative":0.25, "Neutral":0.5, "Positive":0.75, "Very Positive": 1}
  result = pipe(text)[0]['label']
  sentiment_number = map.get(result)
  return sentiment_number


def collect_all_headline_data(keywords:list[str],from_date:str,to_date:str):
  headlines = get_headlines(keywords=keywords, from_date=from_date,to_date=to_date)
  with_sentiment = []
  for headline in headlines:
    text = [headline['title'] + " " + headline['description']]
    sentiment = get_sentiment(text)
    headline['sentiment'] = [sentiment]
    with_sentiment.append(headline)
  df = pd.DataFrame(with_sentiment)
  return df

#Jan 1 to Aug 1st - 679
#Aug 1st to September 1st - 990
#September 1st to Sep 15th - 585
#September 15th to Oct 1st - 679
#Oct 1st to Oct 15th - 388
#Oct 15th to Nov 1st - 846
#Nov 1st - Nov 17th - 1000 (will trim down to Nov15T0000)

def collect_2025_data():
  all_df = pd.DataFrame()
  jan_to_aug = collect_all_headline_data(keywords=['inflation OR cpi'],from_date="2025-01-01T00:00:00.000Z",to_date="2025-08-01T00:00:00.000Z")
  print(len(jan_to_aug))
  all_df = pd.concat([all_df,jan_to_aug])
  aug_to_sep = collect_all_headline_data(keywords=['inflation OR cpi'],from_date="2025-08-01T00:00:00.000Z",to_date="2025-09-01T00:00:00.000Z")
  print(len(aug_to_sep))
  all_df = pd.concat([all_df,aug_to_sep])
  sep_half1 = collect_all_headline_data(keywords=['inflation OR cpi'],from_date="2025-09-01T00:00:00.000Z",to_date="2025-09-15T00:00:00.000Z")
  print(len(sep_half1))
  all_df = pd.concat([all_df,sep_half1])
  sep_half2 = collect_all_headline_data(keywords=['inflation OR cpi'],from_date="2025-09-15T00:00:00.000Z",to_date="2025-10-01T00:00:00.000Z")
  print(len(jan_to_aug))
  all_df = pd.concat([all_df,sep_half2])
  oct_half1 = collect_all_headline_data(keywords=['inflation OR cpi'],from_date="2025-10-01T00:00:00.000Z",to_date="2025-10-15T00:00:00.000Z")
  print(len(oct_half1))
  all_df = pd.concat([all_df,oct_half1])
  oct_half2 = collect_all_headline_data(keywords=['inflation OR cpi'],from_date="2025-10-15T00:00:00.000Z",to_date="2025-11-01T00:00:00.000Z")
  print(len(oct_half2)) 
  all_df = pd.concat([all_df,oct_half2])
  nov_half1 = collect_all_headline_data(keywords=['inflation OR cpi'],from_date="2025-011-01T00:00:00.000Z",to_date="2025-11-17T00:00:00.000Z")
  print(len(nov_half1))
  all_df = pd.concat([all_df,nov_half1])
  all_df.to_csv('all_headline_data.csv')

# print(get_headlines(keywords=keywords,from_date="2025-10-01T00:00:00.000Z",to_date="2025-11-05T00:00:00.000Z"))
# print(get_sentiment('Fed’s Paulson Backs Two More 2025 Rate Cuts Despite Tariffs Federal Reserve Bank of Philadelphia President Anna Paulson signaled she favors two more quarter-point interest-rate cuts this year, as monetary policy should look through the impact of tariffs in consumer price increases.'))
#print(collect_all_headline_data(keywords=['inflation OR cpi'],from_date="2025-01-01T00:00:00.000Z",to_date="2025-06-01T00:00:00.000Z"))
#print(collect_all_headline_data(keywords=['inflation OR cpi'],from_date="2025-09-01T00:00:00.000Z",to_date="2025-09-15T00:00:00.000Z"))
#print(collect_all_headline_data(keywords=['inflation OR cpi'],from_date="2025-10-15T00:00:00.000Z",to_date="2025-11-01T00:00:00.000Z"))
collect_2025_data()