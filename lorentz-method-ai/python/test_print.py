print("Starting test_print...")
import sys; sys.stdout.flush()

try:
    print("Importing pandas")
    import pandas as pd
    print("Pandas imported")
except:
    pass

try:
    print("Importing lc.pipeline.knn_strategy_parity")
    import lc.pipeline.knn_strategy_parity
    print("imported!")
except Exception as e:
    print(e)
