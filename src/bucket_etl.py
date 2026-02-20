from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src import etl
except ModuleNotFoundError:
    import etl


def create_bucket_probability_df(
    thresholds: dict[str, float],
    df: pd.DataFrame,
    source_col: str = 'timestamp',
) -> pd.DataFrame:
    ordered = [(col, t) for col, t in etl._ordered_threshold_items(thresholds) if col in df.columns]
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

    # Enforce non-increasing cumulative probabilities before differencing.
    surv_vals = np.minimum.accumulate(surv.to_numpy(), axis=1)
    surv = pd.DataFrame(surv_vals, index=surv.index, columns=surv.columns)

    bucket_cols: list[str] = []

    first_col = prob_cols[0]
    left_col_name = f"bucket_le_{etl._sanitize_column_name(first_col)}"
    bucket_df[left_col_name] = (1.0 - surv[first_col]).clip(lower=0, upper=1)
    bucket_cols.append(left_col_name)

    for left_col, right_col in zip(prob_cols[:-1], prob_cols[1:]):
        bucket_col = (
            f"bucket_{etl._sanitize_column_name(left_col)}"
            f"_to_{etl._sanitize_column_name(right_col)}"
        )
        bucket_df[bucket_col] = (surv[left_col] - surv[right_col]).clip(lower=0, upper=1)
        bucket_cols.append(bucket_col)

    last_col = prob_cols[-1]
    right_col_name = f"bucket_gt_{etl._sanitize_column_name(last_col)}"
    bucket_df[right_col_name] = surv[last_col].clip(lower=0, upper=1)
    bucket_cols.append(right_col_name)

    bucket_df['bucket_total_probability'] = bucket_df[bucket_cols].sum(axis=1)
    return bucket_df


def join_bucket_kalshi_data(
    output_file: str = 'data/kalshi_headlines_bucket_joined.csv',
    headlines_file: str = 'data/headlines/daily_headlines.csv',
    market_dirs: dict[str, str] | None = None,
    rolling_window: int = etl.DEFAULT_ROLLING_WINDOW,
) -> pd.DataFrame:
    if market_dirs is None:
        market_dirs = {
            'cpi': 'data/cpi',
            'cpi_core': 'data/cpi_core',
            'cpi_yoy': 'data/cpi_yoy',
        }

    if not Path(headlines_file).exists():
        etl.aggregate_daily_headlines(headlines_file)

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

            thresholds = etl.extract_thresholds(list(df.columns))
            prob_cols = etl._get_probability_columns(df, thresholds)
            for col in prob_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df['timestamp'] = (
                pd.to_datetime(df['timestamp'], errors='coerce')
                .dt.tz_localize(None)
                .dt.normalize()
            )
            df = etl.calculate_expected_cpi(thresholds, df)
            df = etl.calculate_implied_volatility(thresholds, df)
            df = etl.calculate_rolling_volatility(thresholds, df, window=rolling_window)
            df = etl.calculate_rolling_change(thresholds, df, window=rolling_window)
            df = etl.calculate_slope_features(thresholds, df)
            df = etl.calculate_tail_risk_measures(thresholds, df)
            df = etl.calculate_threshold_spread(thresholds, df)

            bucket_df = create_bucket_probability_df(thresholds, df, source_col='timestamp')
            df = df.merge(bucket_df, on='timestamp', how='left')
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

join_bucket_kalshi_data()
