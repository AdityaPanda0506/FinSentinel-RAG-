import json
import pandas as pd
from ml.train import load_artifacts
from sklearn.metrics import classification_report, confusion_matrix

# Load model and features
model, le, feat_cols = load_artifacts()
df = pd.read_csv('data/features.csv', parse_dates=['date'])
df = df.sort_values('date')

# Use last 15% as test set
dates   = df['date'].sort_values().unique()
cutoff  = dates[int(len(dates) * 0.85)]
test_df = df[df['date'] >= cutoff].copy()
test_df[feat_cols] = test_df[feat_cols].fillna(0)

X_test = test_df[feat_cols]
y_test = test_df['signal']

# Predictions
y_enc  = le.transform(y_test)
y_pred = model.predict(X_test)

print('=' * 60)
print('FINSENTINEL — MODEL PERFORMANCE REPORT')
print('=' * 60)
print('Test period  :', test_df['date'].min().date(), 'to', test_df['date'].max().date())
print('Test samples :', len(test_df))
print('Tickers      :', test_df['ticker'].nunique())
print()

# Accuracy
acc      = (y_pred == y_enc).mean()
dir_mask = y_test.isin(['buy', 'sell'])
dir_acc  = (le.inverse_transform(y_pred)[dir_mask] == y_test.values[dir_mask]).mean()
print('Overall Accuracy     :', round(acc * 100, 1), '%')
print('Directional Accuracy :', round(dir_acc * 100, 1), '% (buy/sell only)')
print()

# Per class
print('Per-Class Breakdown:')
print(classification_report(y_enc, y_pred, target_names=le.classes_))

# Confusion matrix
print('Confusion Matrix:')
cm    = confusion_matrix(y_enc, y_pred)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print(cm_df)
print()

# Backtest
try:
    with open('data/backtest_results.json') as f:
        bt = json.load(f)
    strat = bt.get('strategy', {})
    bench = bt.get('benchmark', {})
    print('Backtest Performance:')
    print('  Total Return  :', round(strat.get('total_return', 0) * 100, 1), '%')
    print('  Ann. Return   :', round(strat.get('annualised_return', 0) * 100, 1), '%')
    print('  Sharpe Ratio  :', round(strat.get('sharpe_ratio', 0), 3))
    print('  Sortino Ratio :', round(strat.get('sortino_ratio', 0), 3))
    print('  Max Drawdown  :', round(strat.get('max_drawdown', 0) * 100, 1), '%')
    print('  Win Rate      :', round(strat.get('win_rate', 0) * 100, 1), '%')
    print('  Total Trades  :', strat.get('total_trades', 0))
    print('  Calmar Ratio  :', round(strat.get('calmar_ratio', 0), 3))
    print()
    print('  vs Buy & Hold :', round(bench.get('total_return', 0) * 100, 1), '%')
    print('  Alpha         :', round(bt.get('alpha', 0) * 100, 1), '%')
    print()
    print('=' * 60)
    print('RESUME LINE:')
    print('Accuracy:', round(acc * 100, 1), '%',
          '| Directional:', round(dir_acc * 100, 1), '%',
          '| Win Rate:', round(strat.get('win_rate', 0) * 100, 1), '%',
          '| Sharpe:', round(strat.get('sharpe_ratio', 0), 2))
    print('=' * 60)
except Exception as e:
    print('Backtest error:', e)