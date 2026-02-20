from pathlib import Path
import re

import numpy as np
import pandas as pd

# Calculates DAILY LEVEL DATA: date, daily article totals, avg sentiment, kalshi market data
DEFAULT_ROLLING_WINDOW = 3
test_cols = [
    'timestamp', 'Above -0.1%', 'Above 0.0%', 'Above 0.1%', 'Above 0.2%',
    'Above 0.3%', 'Above 0.4%', 'Above 0.5%'
]
test_df = pd.DataFrame([
    {
        "timestamp": "2025-04-01T00:00:00Z",
        "Above -0.1%": 99,
        "Above 0.0%": 97,
        "Above 0.1%": 88,
        "Above 0.2%": 62,
        "Above 0.3%": 28,
        "Above 0.4%": 11,
        "Above 0.5%": 3,
    },
    {
        "timestamp": "2025-04-02T00:00:00Z",
        "Above -0.1%": 99,
        "Above 0.0%": 96,
        "Above 0.1%": 82,
        "Above 0.2%": 55,
        "Above 0.3%": 21,
        "Above 0.4%": 8,
        "Above 0.5%": 2,
    },
    {
        "timestamp": "2025-04-03T00:00:00Z",
        "Above -0.1%": 99,
        "Above 0.0%": 94,
        "Above 0.1%": 79,
        "Above 0.2%": 51,
        "Above 0.3%": 20,
        "Above 0.4%": 7,
        "Above 0.5%": 2,
    }
])


# Aggregate daily article count, mean sentiment, median sentiment, variance, % pos/neg, min/max sentiment
def aggregate_daily_headlines(output_file: str):
    all_headlines = pd.read_csv('data/headlines/all_headline_data.csv')
    if "Unnamed: 0" in all_headlines.columns:
        all_headlines = all_headlines.drop(columns=["Unnamed: 0"])
    all_headlines['date'] = all_headlines['date'].astype(str).str.split('T').str[0]
    all_headlines['sentiment'] = (
        all_headlines['sentiment'].astype(str).str.strip("[]").astype(float)
    )
    daily_headlines = all_headlines.groupby('date', as_index=False).agg(
        headline_count=('id', 'count'),
        mean_sentiment=('sentiment', 'mean'),
        median_sentiment=('sentiment', 'median'),
        sentiment_std=('sentiment', 'std'),
        sentiment_min=('sentiment', 'min'),
        sentiment_max=('sentiment', 'max'),
        sentiment_neg_pct=('sentiment', lambda x: (x < 0.5).sum()),
        sentiment_pos_pct=('sentiment', lambda x: (x > 0.5).sum()),
        unique_sources_count=("source", lambda x: len(set(x))),
        unique_sources=("source", lambda x: list(set(x))),
    )
    daily_headlines = daily_headlines.fillna(0)
    daily_headlines['date'] = pd.to_datetime(daily_headlines['date'], format='%Y-%m-%d')
    daily_headlines['day_of_week'] = daily_headlines['date'].dt.weekday
    daily_headlines.to_csv(output_file, mode='w', index=False)
    return daily_headlines


# Helper function to extract probability thresholds from kalshi data (arg: list of dataframe columns)
def extract_thresholds(columns: list[str]) -> dict[str, float]:
    thresholds_dict = {}
    for c in columns:
        # remove the % so appending float works
        s = c.replace('%', "")
        for t in s.split():
            try:
                thresholds_dict[c] = float(t)
            except ValueError:
                pass
    return thresholds_dict


def _sanitize_column_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def _ordered_threshold_items(thresholds: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(thresholds.items(), key=lambda item: item[1])


def _get_probability_columns(df: pd.DataFrame, thresholds: dict[str, float]) -> list[str]:
    ordered = _ordered_threshold_items(thresholds)
    return [col for col, _ in ordered if col in df.columns]


def create_bucket_probability_df(
    thresholds: dict[str, float],
    df: pd.DataFrame,
    source_col: str = 'timestamp',
) -> pd.DataFrame:
    ordered = [(col, t) for col, t in _ordered_threshold_items(thresholds) if col in df.columns]
    bucket_df = pd.DataFrame(index=df.index)

    if source_col in df.columns:
        bucket_df[source_col] = df[source_col]

    if not ordered:
        bucket_df['bucket_total_probability'] = 0.0
        return bucket_df

    prob_cols = [col for col, _ in ordered]
    surv = (
        df[prob_cols]
        .apply(pd.to_numeric, errors='coerce')
        .fillna(0)
        .clip(lower=0, upper=100)
        / 100.0
    )

    # Kalshi "Above x%" columns are cumulative probabilities and should be non-increasing.
    # Enforce monotonicity to avoid negative bucket mass from noisy quotes.
    surv_vals = np.minimum.accumulate(surv.to_numpy(), axis=1)
    surv = pd.DataFrame(surv_vals, index=surv.index, columns=surv.columns)

    bucket_cols: list[str] = []

    first_col = prob_cols[0]
    left_col_name = f"bucket_le_{_sanitize_column_name(first_col)}"
    bucket_df[left_col_name] = (1.0 - surv[first_col]).clip(lower=0, upper=1)
    bucket_cols.append(left_col_name)

    for left_col, right_col in zip(prob_cols[:-1], prob_cols[1:]):
        bucket_col = (
            f"bucket_{_sanitize_column_name(left_col)}"
            f"_to_{_sanitize_column_name(right_col)}"
        )
        bucket_df[bucket_col] = (surv[left_col] - surv[right_col]).clip(lower=0, upper=1)
        bucket_cols.append(bucket_col)

    last_col = prob_cols[-1]
    right_col_name = f"bucket_gt_{_sanitize_column_name(last_col)}"
    bucket_df[right_col_name] = surv[last_col].clip(lower=0, upper=1)
    bucket_cols.append(right_col_name)

    bucket_df['bucket_total_probability'] = bucket_df[bucket_cols].sum(axis=1)
    return bucket_df


# Calculates and adds expected cpi column to dataframe
def calculate_expected_cpi(thresholds: dict, df: pd.DataFrame) -> pd.DataFrame:
    prob_cols = _get_probability_columns(df, thresholds)
    if not prob_cols:
        df['expected_cpi'] = 0.0
        return df

    probs = df[prob_cols].fillna(0).to_numpy() / 100.0
    threshold_values = [thresholds[col] for col in prob_cols]
    deltas = np.diff(threshold_values)
    if len(deltas) == 0:
        df['expected_cpi'] = probs[:, 0] * threshold_values[0]
        return df

    # all deltas are the same
    if np.allclose(deltas, deltas[0]):
        delta = deltas[0]
        df['expected_cpi'] = probs.sum(axis=1) * delta
        return df

    # non-uniform spacing (different deltas) so pad the last delta
    deltas_full = np.append(deltas, deltas[-1])
    df['expected_cpi'] = (probs * deltas_full).sum(axis=1)
    return df


# Calculates and adds implied volatility to dataframe
def calculate_implied_volatility(thresholds: dict, df: pd.DataFrame) -> pd.DataFrame:
    prob_cols = [c for c in _get_probability_columns(df, thresholds) if thresholds[c] >= 0.0]
    if not prob_cols:
        df['implied_volatility'] = 0.0
        return df

    probs = df[prob_cols].fillna(0).to_numpy() / 100.0
    th_vals = [thresholds[c] for c in prob_cols]
    deltas = np.diff(th_vals)
    if len(deltas) == 0:
        df['implied_volatility'] = 0.0
        return df

    delta = deltas[0]
    # E[X] ~= sum_k S(t_k) * dt
    ex = probs.sum(axis=1) * delta
    # E[X^2] ~= sum_k 2 * t_k * S(t_k) * dt
    ex2 = (2 * (probs * th_vals)).sum(axis=1) * delta
    # Var(X)=E[X^2]-(E[X])^2
    var = ex2 - (ex ** 2)
    # Clip to avoid floating point error leading to negative number sqrt
    var = np.clip(var, 0, None)
    df['implied_volatility'] = np.sqrt(var)
    return df


def calculate_rolling_volatility(
    thresholds: dict, df: pd.DataFrame, window: int = DEFAULT_ROLLING_WINDOW
) -> pd.DataFrame:
    prob_cols = _get_probability_columns(df, thresholds)
    for col in prob_cols:
        feature_col = f"{_sanitize_column_name(col)}_rolling_std"
        df[feature_col] = (
            pd.to_numeric(df[col], errors='coerce')
            .rolling(window=window, min_periods=1)
            .std()
            .fillna(0)
        )
    return df


def calculate_rolling_change(
    thresholds: dict, df: pd.DataFrame, window: int = DEFAULT_ROLLING_WINDOW
) -> pd.DataFrame:
    prob_cols = _get_probability_columns(df, thresholds)
    for col in prob_cols:
        feature_col = f"{_sanitize_column_name(col)}_rolling_change"
        change = pd.to_numeric(df[col], errors='coerce').diff()
        df[feature_col] = change.rolling(window=window, min_periods=1).sum().fillna(0)
    return df


def calculate_slope_features(thresholds: dict, df: pd.DataFrame) -> pd.DataFrame:
    ordered = _ordered_threshold_items(thresholds)
    ordered_cols = [col for col, _ in ordered if col in df.columns]
    for i in range(len(ordered_cols) - 1):
        col_left = ordered_cols[i]
        col_right = ordered_cols[i + 1]
        feature_col = f"slope_{_sanitize_column_name(col_left)}_to_{_sanitize_column_name(col_right)}"
        df[feature_col] = (
            pd.to_numeric(df[col_left], errors='coerce')
            - pd.to_numeric(df[col_right], errors='coerce')
        )
    return df


def calculate_tail_risk_measures(thresholds: dict, df: pd.DataFrame) -> pd.DataFrame:
    ordered = _ordered_threshold_items(thresholds)
    ordered_existing = [(col, v) for col, v in ordered if col in df.columns]
    non_neg = [(col, v) for col, v in ordered_existing if v >= 0.0]
    right_tail_cols = [col for col, v in ordered_existing if v >= 0.4]

    baseline_col = None
    for col, v in non_neg:
        if np.isclose(v, 0.0):
            baseline_col = col
            break
    if baseline_col is None and non_neg:
        baseline_col = non_neg[0][0]

    if right_tail_cols:
        df['right_tail_sum'] = df[right_tail_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
    else:
        df['right_tail_sum'] = 0.0

    if baseline_col is not None:
        baseline = pd.to_numeric(df[baseline_col], errors='coerce').replace(0, np.nan)
        df['right_tail_share'] = (df['right_tail_sum'] / baseline).fillna(0)
    else:
        df['right_tail_share'] = 0.0

    neg_cols = [col for col, v in ordered_existing if v < 0.0]
    if neg_cols and baseline_col is not None:
        closest_neg_col = neg_cols[-1]
        df['left_tail_gap'] = (
            pd.to_numeric(df[closest_neg_col], errors='coerce')
            - pd.to_numeric(df[baseline_col], errors='coerce')
        ).clip(lower=0)
    else:
        df['left_tail_gap'] = 0.0

    return df


def calculate_threshold_spread(thresholds: dict, df: pd.DataFrame) -> pd.DataFrame:
    prob_cols = _get_probability_columns(df, thresholds)
    if not prob_cols:
        df['threshold_spread'] = 0.0
        return df
    probs = df[prob_cols].apply(pd.to_numeric, errors='coerce')
    df['threshold_spread'] = probs.max(axis=1) - probs.min(axis=1)
    return df


# Creates some features based on kalshi data before exploration
# Joins cpi, cpi_yoy, and cpi_core kalshi market data to headline data
# date, market, headline data
def join_kalshi_data(
    output_file: str = 'data/headlines/kalshi_headlines_joined.csv',
    headlines_file: str = 'data/headlines/daily_headlines.csv',
    market_dirs: dict[str, str] | None = None,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
) -> pd.DataFrame:
    if market_dirs is None:
        market_dirs = {
            'cpi': 'data/cpi',
            'cpi_core': 'data/cpi_core',
            'cpi_yoy': 'data/cpi_yoy',
        }

    if not Path(headlines_file).exists():
        aggregate_daily_headlines(headlines_file)

    daily_headlines = pd.read_csv(headlines_file)
    if "Unnamed: 0" in daily_headlines.columns:
        daily_headlines = daily_headlines.drop(columns=["Unnamed: 0"])
    daily_headlines['date'] = (
        pd.to_datetime(daily_headlines['date'], errors='coerce')
        .dt.tz_localize(None)
        .dt.normalize()
    )

    market_feature_frames = []
    for market, folder in market_dirs.items():
        market_path = Path(folder)
        csv_files = sorted(market_path.glob("*.csv"))
        if not csv_files:
            continue

        month_frames = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            if 'timestamp' not in df.columns:
                continue

            thresholds = extract_thresholds(list(df.columns))
            prob_cols = _get_probability_columns(df, thresholds)
            for col in prob_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df['timestamp'] = (
                pd.to_datetime(df['timestamp'], errors='coerce')
                .dt.tz_localize(None)
                .dt.normalize()
            )
            df = calculate_expected_cpi(thresholds, df)
            df = calculate_implied_volatility(thresholds, df)
            df = calculate_rolling_volatility(thresholds, df, window=rolling_window)
            df = calculate_rolling_change(thresholds, df, window=rolling_window)
            df = calculate_slope_features(thresholds, df)
            df = calculate_tail_risk_measures(thresholds, df)
            df = calculate_threshold_spread(thresholds, df)
            df = df.rename(columns={'timestamp': 'date'})
            month_frames.append(df)

        if not month_frames:
            continue

        market_df = pd.concat(month_frames, ignore_index=True)
        numeric_cols = market_df.select_dtypes(include=[np.number]).columns.tolist()
        keep_cols = ['date'] + numeric_cols
        market_daily = (
            market_df[keep_cols]
            .groupby('date', as_index=False)
            .mean(numeric_only=True)
            .sort_values('date')
        )
        rename_map = {col: f"{market}_{col}" for col in market_daily.columns if col != 'date'}
        market_daily = market_daily.rename(columns=rename_map)
        market_feature_frames.append(market_daily)

    if market_feature_frames:
        kalshi_joined = market_feature_frames[0]
        for market_df in market_feature_frames[1:]:
            kalshi_joined = kalshi_joined.merge(market_df, on='date', how='outer')
    else:
        kalshi_joined = pd.DataFrame({'date': daily_headlines['date']})

    joined = daily_headlines.merge(kalshi_joined, on='date', how='left').sort_values('date')
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output_file, index=False)
    return joined


join_kalshi_data().to_csv('joined_data.csv', index= False)