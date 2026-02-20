Analyzing how daily inflation-related news volume and headline sentiment correlate with CPI prediction market volatility on Kalshi between Jan 2025 and Nov 2025.

Just a side project exploring a curiousity.

Data from GNews API and Kalshi.

## Significant Findings
There's much more to explore, and the data quality isn't the best, but here's a short list of some of my findings.
- Headline count has the strongest relationship with CPI YoY implied volatility (positive, modest strength).
- Headline spikes and volatility spikes rarely happen on the exact same day; overlap is limited.
- Core CPI implied volatility shows a weak negative relationship with headline volume in this sample.
- Sentiment features are generally weaker than headline volume for explaining volatility moves.
- Bucket-based analysis shows similar conclusions: strongest signals are tied to headline intensity, not average sentiment.
- Calendar effects appear in both series, but their weekly/monthly cycles are only partially aligned.
- Kalshi activity/volume tends to spike during weekends; news activity is strongest on weekdays (Friday specifically), generally
