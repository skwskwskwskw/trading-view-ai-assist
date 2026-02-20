import time
from traceback import print_exc

print("Starting checks...")
try:
    from lc.pipeline.io import load_ohlcv_csv
    from lc.pipeline.knn_strategy_parity import KNNParityConfig, run_knn_parity
    print("Imports OK.")
except Exception as e:
    print("Import error:")
    print_exc()

import sys
file_path = "MYX_FCPO1!, 5.csv"

try:
    print(f"Loading {file_path}")
    df = load_ohlcv_csv(file_path)
    print(f"Loaded {len(df)} rows.")
except Exception as e:
    print("Load error:")
    print_exc()

try:
    print("Running config init...")
    cfg = KNNParityConfig()
    print("Running run_knn_parity...")
    import multiprocessing
    # Run the function, but print how far it gets
    # I'll just run it directly
    trace, ledger = run_knn_parity(df.head(200), cfg)
    print("Success on 200 rows! Length of trace:", len(trace))
except Exception as e:
    print("Run error:")
    print_exc()
