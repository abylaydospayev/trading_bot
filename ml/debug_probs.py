import pandas as pd, numpy as np
from ml.infer import predict_entry_prob

model_path = 'ml/models/mlp_thr005.pkl'
fp = 'ml/data/dataset_BTC_USD_2024-10-01_2025-04-14_15m.csv'

df = pd.read_csv(fp, index_col=0, parse_dates=True)
probs = []
for i in range(min(2000, len(df))):
    row = df.iloc[i].copy()
    p = predict_entry_prob(row, model_path)
    probs.append(p)

print('count', len(probs), 'min', np.min(probs), 'max', np.max(probs), 'mean', float(np.mean(probs)))
print('top5', sorted(probs)[-5:])
