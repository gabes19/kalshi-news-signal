#Calculates DAILY LEVEL DATA: date, daily article totals, avg sentiment
#Joins headlines and kalshi data and cleans it
import pandas as pd
all_headlines = pd.read_csv('data/headlines/all_headline_data.csv')
if "Unnamed: 0" in all_headlines.columns:
    all_headlines = all_headlines.drop(columns=["Unnamed: 0"])

#Aggregate daily article count, mean sentiment, median sentiment, variance, % pos/neg, min/max sentiment
def aggregate_daily_headlines(output_file:str):
    all_headlines['date'] = all_headlines['date'].astype(str).str.split('T').str[0]
    all_headlines['sentiment'] = (
    all_headlines['sentiment'].astype(str).str.strip("[]").astype(float))
    daily_headlines = all_headlines.groupby('date').agg(
        headline_count = ('id', 'count'),
        mean_sentiment = ('sentiment', 'mean'),
        median_sentiment = ('sentiment', 'median'),
        sentiment_std = ('sentiment', 'std'),
        sentiment_min = ('sentiment', 'min'),
        sentiment_max = ('sentiment', 'max'),
        sentiment_neg_pct = ('sentiment', lambda x: (x < 0.5).sum()),
        sentiment_pos_pct = ('sentiment', lambda x: (x > 0.5).sum()),
        unique_sources_count = ("source", lambda x: len(set(x))),
        unique_sources = ("source", lambda x: list(set(x)))
    )
    daily_headlines = daily_headlines.fillna(0)
    print(daily_headlines.head())
    daily_headlines.to_csv(output_file, mode='w')
    print(f"File saved as {output_file}")




aggregate_daily_headlines('data/headlines/daily_headlines.csv')
