import pandas as pd
import re
#Calculates DAILY LEVEL DATA: date, daily article totals, avg sentiment, kalshi market data


#Aggregate daily article count, mean sentiment, median sentiment, variance, % pos/neg, min/max sentiment
def aggregate_daily_headlines(output_file:str):
    all_headlines = pd.read_csv('data/headlines/all_headline_data.csv')
    if "Unnamed: 0" in all_headlines.columns:
        all_headlines = all_headlines.drop(columns=["Unnamed: 0"])
    all_headlines['date'] = all_headlines['date'].astype(str).str.split('T').str[0]
    all_headlines['sentiment'] = (
    all_headlines['sentiment'].astype(str).str.strip("[]").astype(float))
    daily_headlines = all_headlines.groupby('date', as_index=False).agg(
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
    daily_headlines['date'] = pd.to_datetime(daily_headlines['date'], format='%Y-%m-%d')
    daily_headlines['day_of_week'] = daily_headlines['date'].dt.weekday
    print(daily_headlines.head())
    daily_headlines.to_csv(output_file, mode='w')
    print(f"File saved as {output_file}")
    return daily_headlines


#Helper function to extract probability thresholds from kalshi data (arg: list of dataframe columns)
#TODO: modify to return dict with key as column name and value as threshold
def extract_thresholds(columns:list[str]) -> dict[str,float]:
    thresholds_dict = {}
    for c in columns:
        #remove the % so appending float works
        s = c.replace('%',"")
        for t in s.split():
            try:
                thresholds_dict[c] =float(t)
            except:
                ValueError
                pass
    return thresholds_dict

#TODO:
def calculate_expected_cpi():
    pass


    

#Creates some features based on kalshi data before exploration
#Joins cpi, cpi_yoy, and cpi_core kalshi market data to headline data
#date, market, headline data
#TODO:
def join_kalshi_data():
    kalshi_df = pd.DataFrame()
    months = ['jan','feb','mar','apr','may','jun','jul','aug','sep''oct','nov']
    markets = ['cpi','cpicore','cpiyoy']
    for market in markets:
        for month in months:
            df = pd.read_csv(f'data/{market}/kalshi-price-history-25{month}-day')
            df['timestamp'] = df['timestamp'].astype(str).str.split('T').str[0]
            columns = list(df.columns)
            #TODO: Transform into dict to map column names and thresholds
            thresholds = extract_thresholds(columns)
            #TODO: Calculate expected_cpi (market-implied mean)
            #TODO: Calculate daily CPI volatility (std = implied vol)
            #TODO: Calculate rolling prob volatility (std) for each threshold
            #TODO: Calculate rolling total change for each threshold
            #TODO: Calculate slope features (slope_01 = P>0.0 - P>0.1, etc.)
            #TODO: Calculate tail risk measures (right_tail_share (P>0.4+P>0.5+P>0.6) =  right_tail / P>0.0, e.g.)
            #TODO: Calculate spread between thresholds
            df = df.rename(columns={'timestamp':'date'})

print(extract_thresholds(['timestamp','Above -0.1%','Above 0.0%','Above 0.1%','Above 0.2%', 'Above 0.3%', 'Above 0.4%', 'Above 0.5%']))
#aggregate_daily_headlines('data/headlines/daily_headlines.csv')
