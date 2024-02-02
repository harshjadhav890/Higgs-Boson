from modelling import preprocess, fit_score
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


df = pd.read_parquet('data/preprocessed_data.parquet')
X, y = preprocess(df)
model, metrics = fit_score(X, y)

for key, value in metrics.items():
    print(f"{key}:{value:.4f}")

with open("results.txt", 'w') as outfile:
    outfile.write("Model: XGBoost\n")
    outfile.write("Metrics:\n")
    for key, value in metrics.items():
        outfile.write(f"\t{key}: {value:.4f}\n")

