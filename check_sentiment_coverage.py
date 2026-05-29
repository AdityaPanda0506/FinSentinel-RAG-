import pandas as pd

df = pd.read_csv('data/features.csv', parse_dates=['date'])

print("Sentiment coverage by year:")
df['year'] = df['date'].dt.year
df['has_sentiment'] = df['sentiment_score'] != 0

coverage = df.groupby('year').agg(
    total_rows   = ('sentiment_score', 'count'),
    with_sentiment = ('has_sentiment', 'sum'),
).assign(coverage_pct=lambda x: (x['with_sentiment'] / x['total_rows'] * 100).round(1))

print(coverage)
print()
print("Overall sentiment coverage:", round(df['has_sentiment'].mean() * 100, 1), "%")