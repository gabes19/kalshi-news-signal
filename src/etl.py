import pandas as pd
import numpy as np
#Calculates DAILY LEVEL DATA: date, daily article totals, avg sentiment, kalshi market data
test_cols = ['timestamp','Above -0.1%','Above 0.0%','Above 0.1%','Above 0.2%', 'Above 0.3%', 'Above 0.4%', 'Above 0.5%']
test_df = pd.DataFrame([
    {
        "timestamp": "2025-04-01T00:00:00Z",
        "Above -0.1%": 99,
        "Above 0.0%": 97,
        "Above 0.1%": 88,
        "Above 0.2%": 62,
        "Above 0.3%": 28,
        "Above 0.4%": 11,
        "Above 0.5%": 3
    },
    {
        "timestamp": "2025-04-02T00:00:00Z",
        "Above -0.1%": 99,
        "Above 0.0%": 96,
        "Above 0.1%": 82,
        "Above 0.2%": 55,
        "Above 0.3%": 21,
        "Above 0.4%": 8,
        "Above 0.5%": 2
    },
    {
        "timestamp": "2025-04-03T00:00:00Z",
        "Above -0.1%": 99,
        "Above 0.0%": 94,
        "Above 0.1%": 79,
        "Above 0.2%": 51,
        "Above 0.3%": 20,
        "Above 0.4%": 7,
        "Above 0.5%": 2
    }
])

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


#Calculates and adds expected cpi column to dataframe
def calculate_expected_cpi(thresholds:dict,df:pd.DataFrame)-> pd.DataFrame:
    prob_cols = list(thresholds.keys())
    probs = df[prob_cols].to_numpy() / 100.0
    deltas = np.diff(list(thresholds.values()))
    #all deltas are the same
    if len(deltas) > 0 and np.allclose(deltas, deltas[0]):
        delta = deltas[0]
        df['expected_cpi'] = probs.sum(axis=1) * delta
        return df
    #non-uniform spacing (different deltas) so pad the last delta
    deltas_full = np.append(deltas,deltas[-1])
    df['expected_cpi'] = (probs * deltas_full).sum(axis = 1)
    return df

#TODO: Test this function
#Calculates and adds imolied volatility to dataframe
def calculate_implied_volatility(thresholds:dict, df:pd.DataFrame) -> pd.DataFrame:
    prob_cols = list(thresholds.keys())
    probs = df[prob_cols].to_numpy() / 100.0
    #assume equal deltas
    deltas = np.diff(list(thresholds.values()))
    if len(deltas) == 0:
        df['implied_volatility'] = 0.0
        return df
    delta = deltas[0]
    #E[X] ≈ sum_k S(t_k) * Δt
    EX = probs.sum(axis=1) * delta
    # E[X^2] ≈ sum_k 2 * t_k * S(t_k) * Δt
    EX2 = (2*(probs*thresholds.values())).sum(axis=1) * delta
    #Var(X)=E[X^22]−(E[X])^2
    var = EX2 - EX ** 2
    #Clip to avoid floating point error leading to negative number sqrt
    var = np.clip(var,0,None)
    df['implied_volatility'] = np.sqrt(var)
    return df

    

#Creates some features based on kalshi data before exploration
#Joins cpi, cpi_yoy, and cpi_core kalshi market data to headline data
#date, market, headline data
#TODO:
def join_kalshi_data():
    kalshi_dfs = []
    months = ['jan','feb','mar','apr','may','jun','jul','aug','sep''oct','nov']
    markets = ['cpi','cpicore','cpiyoy']
    for market in markets:
        for month in months:
            df = pd.read_csv(f'data/{market}/kalshi-price-history-25{month}-day')
            df['timestamp'] = df['timestamp'].astype(str).str.split('T').str[0]
            columns = list(df.columns)
            thresholds = extract_thresholds(columns)
            df = calculate_expected_cpi(thresholds,df)
            df = calculate_implied_volatility(thresholds,df)
            #TODO: Calculate rolling prob volatility (std) for each threshold
            #TODO: Calculate rolling total change for each threshold
            #TODO: Calculate slope features (slope_01 = P>0.0 - P>0.1, etc.)
            #TODO: Calculate tail risk measures (right_tail_share (P>0.4+P>0.5+P>0.6) =  right_tail / P>0.0, e.g.)
            #TODO: Calculate spread between thresholds
            df = df.rename(columns={'timestamp':'date'})

print(calculate_expected_cpi(extract_thresholds(test_cols),test_df))
#print(extract_thresholds(test_cols))
#aggregate_daily_headlines('data/headlines/daily_headlines.csv')
