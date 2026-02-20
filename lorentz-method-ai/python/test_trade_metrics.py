import pandas as pd
from lc.pipeline.trade_metrics import extract_trade_records

# Sample data: start a long, then start short immediately (without explicit end_long)
data = {
    'close': [100, 110, 105, 90],
    'start_long': [True, False, False, False],
    'end_long': [False, False, False, False],
    'start_short': [False, True, False, False],
    'end_short': [False, False, False, False]
}
df = pd.DataFrame(data)

try:
    trades = extract_trade_records(df)
    print("Extracted trades:")
    print(trades)
except Exception as e:
    print(f"Error: {e}")
