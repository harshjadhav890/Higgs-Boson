from modelling import preprocess, fit_score
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import json


df = pd.read_parquet('data/preprocessed_data.parquet')
X, y = preprocess(df)
model, metrics, cm = fit_score(X, y)

with open('metrics.json', "w") as json_file:
    json.dump(metrics, json_file)

# for key, value in metrics.items():
#     print(f"{key}:{value:.4f}")

# with open("results.txt", 'w') as outfile:
#     outfile.write("Model: XGBoost\n")
#     outfile.write("Metrics:\n")
#     for key, value in metrics.items():
#         outfile.write(f"\t{key}: {value:.4f}\n")

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Adjust font scale for better readability
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False,
            xticklabels=['Predicted Background', 'Predicted Signal'],
            yticklabels=['Actual Background', 'Actual Signal'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
# plt.show()

# Save plot as PNG
plt.savefig('images/confusion_matrix.png')