import pandas as pd
import pytest

from src import etl


def _build_test_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "timestamp": "2025-04-01T00:00:00Z",
            "Above -0.1%": 99.0,
            "Above 0.0%": 97.0,
            "Above 0.1%": 88.0,
            "Above 0.2%": 62.0,
            "Above 0.3%": 28.0,
            "Above 0.4%": 11.0,
            "Above 0.5%": 3.0,
        },
        {
            "timestamp": "2025-04-02T00:00:00Z",
            "Above -0.1%": 99.0,
            "Above 0.0%": 96.0,
            "Above 0.1%": 82.0,
            "Above 0.2%": 55.0,
            "Above 0.3%": 21.0,
            "Above 0.4%": 8.0,
            "Above 0.5%": 2.0,
        },
        {
            "timestamp": "2025-04-03T00:00:00Z",
            "Above -0.1%": 99.0,
            "Above 0.0%": 94.0,
            "Above 0.1%": 79.0,
            "Above 0.2%": 51.0,
            "Above 0.3%": 20.0,
            "Above 0.4%": 7.0,
            "Above 0.5%": 2.0,
        },
    ])


def test_extract_thresholds():
    thresholds = etl.extract_thresholds(etl.test_cols)
    assert thresholds["Above -0.1%"] == -0.1
    assert thresholds["Above 0.0%"] == 0.0
    assert thresholds["Above 0.5%"] == 0.5


def test_calculate_expected_cpi():
    df = _build_test_df()
    thresholds = etl.extract_thresholds(df.columns.tolist())
    result = etl.calculate_expected_cpi(thresholds, df.copy())

    expected_first = (99 + 97 + 88 + 62 + 28 + 11 + 3) / 100.0 * 0.1
    assert result["expected_cpi"].iloc[0] == pytest.approx(expected_first)


def test_calculate_implied_volatility_non_negative():
    df = _build_test_df()
    thresholds = etl.extract_thresholds(df.columns.tolist())
    result = etl.calculate_implied_volatility(thresholds, df.copy())

    assert (result["implied_volatility"] >= 0).all()


def test_calculate_rolling_volatility_and_change():
    df = _build_test_df()
    thresholds = etl.extract_thresholds(df.columns.tolist())

    result = etl.calculate_rolling_volatility(thresholds, df.copy(), window=3)
    result = etl.calculate_rolling_change(thresholds, result, window=3)

    std_col = "above_0_0_rolling_std"
    chg_col = "above_0_0_rolling_change"
    assert std_col in result.columns
    assert chg_col in result.columns

    # 97, 96, 94 -> rolling std with ddof=1 at row 3
    assert result[std_col].iloc[0] == 0
    assert result[std_col].iloc[1] == pytest.approx(0.707106, rel=1e-4)
    assert result[std_col].iloc[2] == pytest.approx(1.527525, rel=1e-4)

    # diffs: nan, -1, -2 -> rolling sums: 0, -1, -3
    assert result[chg_col].tolist() == pytest.approx([0.0, -1.0, -3.0])


def test_calculate_slope_tail_and_spread_features():
    df = _build_test_df()
    thresholds = etl.extract_thresholds(df.columns.tolist())

    result = etl.calculate_slope_features(thresholds, df.copy())
    result = etl.calculate_tail_risk_measures(thresholds, result)
    result = etl.calculate_threshold_spread(thresholds, result)

    slope_col = "slope_above_0_0_to_above_0_1"
    assert slope_col in result.columns
    assert result[slope_col].iloc[0] == pytest.approx(9.0)

    assert result["right_tail_sum"].iloc[0] == pytest.approx(14.0)
    assert result["right_tail_share"].iloc[0] == pytest.approx(14.0 / 97.0)
    assert result["left_tail_gap"].iloc[0] == pytest.approx(2.0)
    assert result["threshold_spread"].iloc[0] == pytest.approx(96.0)


def test_join_kalshi_data_writes_csv(tmp_path):
    for market in ["cpi", "cpi_core", "cpi_yoy"]:
        (tmp_path / market).mkdir(parents=True, exist_ok=True)

    cpi_df = _build_test_df()
    cpi_core_df = cpi_df.drop(columns=["Above -0.1%"])
    cpi_yoy_df = pd.DataFrame([
        {
            "timestamp": "2025-04-01T00:00:00Z",
            "Above 2.0%": 80.0,
            "Above 2.1%": 70.0,
            "Above 2.2%": 60.0,
            "Above 2.3%": 50.0,
            "Above 2.4%": 40.0,
        },
        {
            "timestamp": "2025-04-02T00:00:00Z",
            "Above 2.0%": 82.0,
            "Above 2.1%": 71.0,
            "Above 2.2%": 59.0,
            "Above 2.3%": 49.0,
            "Above 2.4%": 39.0,
        },
    ])

    cpi_df.to_csv(tmp_path / "cpi" / "kalshi-price-history-kxcpi-25apr-day.csv", index=False)
    cpi_core_df.to_csv(tmp_path / "cpi_core" / "kalshi-price-history-kxcpicore-25apr-day.csv", index=False)
    cpi_yoy_df.to_csv(tmp_path / "cpi_yoy" / "kalshi-price-history-kxcpiyoy-25apr-day.csv", index=False)

    headlines = pd.DataFrame([
        {"date": "2025-04-01", "headline_count": 10, "mean_sentiment": 0.5},
        {"date": "2025-04-02", "headline_count": 11, "mean_sentiment": 0.6},
    ])
    headlines_file = tmp_path / "daily_headlines.csv"
    headlines.to_csv(headlines_file, index=False)

    output_file = tmp_path / "joined.csv"
    joined = etl.join_kalshi_data(
        output_file=str(output_file),
        headlines_file=str(headlines_file),
        market_dirs={
            "cpi": str(tmp_path / "cpi"),
            "cpi_core": str(tmp_path / "cpi_core"),
            "cpi_yoy": str(tmp_path / "cpi_yoy"),
        },
    )

    assert output_file.exists()
    assert "headline_count" in joined.columns
    assert "cpi_expected_cpi" in joined.columns
    assert "cpi_core_expected_cpi" in joined.columns
    assert "cpi_yoy_expected_cpi" in joined.columns
