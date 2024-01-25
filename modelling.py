import math
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
# to reset option use: pd.reset_option('max_columns')
warnings.filterwarnings("ignore")

import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, precision_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models

import imblearn
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

# show all columns in a df while using functions like .head()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


## Preprocessing 
df = pd.read_csv('data/training.zip', low_memory=False)

def preprocess(df):
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df['Label'])
    X = df.drop('Label', axis=1)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
    
X_train, X_test, y_train, y_test = preprocess(df)


## Modelling and CV

xgb_model = XGBClassifier(objective='binary:logistic', random_state=42)
xgb_model.fit(X_train, y_train)

precision_scorer = make_scorer(precision_score)
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
# Perform cross-validation and calculate precision score
precision_scores = cross_val_score(xgb_model, X_train, y_train, cv=k_fold, scoring=precision_scorer)

# Print the precision scores for each fold
for i, precision in enumerate(precision_scores):
    print(f'Fold {i+1}: Precision = {precision}')
    
# Print the average precision across all folds
print(f'\nAverage Precision: {precision_scores.mean()}')

# Make predictions on the test data
y_pred = xgb_model.predict(X_test)
# Calculate precision on the test data
precision = precision_score(y_test, y_pred)
print(f'\nTest data Precision: {precision}')

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Print classification report
class_report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(class_report)



