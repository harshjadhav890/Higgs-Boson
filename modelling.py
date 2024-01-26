import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# import argparse
# parser = argparse.ArgumentParser(description='Modelling script')

# parser.add_argument('--model', type=str, default= "ML", help='Choose between ML and DL')
# parser.add_argument('--path', type=str, default='data/training.zip', help='Add dataset path')
# args = parser.parse_args()


def preprocess(df):
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df['Label'])
    X = df.drop(['Label', 'Weight'], axis=1)
    y = df['Label']
    return X, y

def fit_score(X, y):
    """Fit, cross-validate and print metrics

    Args:
        X (pandas.core.frame.DataFrame): Independent variables
        y (pandas.core.frame.DataFrame): Label to predict

    Returns:
        xgboost.sklearn.XGBClassifier: A baseline XGB classifier
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    xgb_model = XGBClassifier(objective='binary:logistic', random_state=42)
    xgb_model.fit(X_train, y_train)

    precision_scorer = make_scorer(precision_score)
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    precision_scores = cross_val_score(xgb_model, X_train, y_train, cv=k_fold, scoring=precision_scorer)
    for i, precision in enumerate(precision_scores):
        print(f'Fold {i+1}: Precision = {precision}')
        
    # Print the average precision across all folds
    print(f'\nAverage Precision: {precision_scores.mean()}')

    # Make predictions on the test data
    y_pred = xgb_model.predict(X_test)
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
    
    return xgb_model

def save(model, path):
    with open(path, 'wb') as model_file:
        pickle.dump(model, model_file)


df = pd.read_csv('data/training.zip', low_memory=False)
X, y = preprocess(df)
model = fit_score(X, y)
save(model, 'models/Baseline_XGB.pkl')

#-----------------------------------------------------------------------------------------------------#

# Here we chose 6 variables that influence the decisions of the model the most and created a model
# only with the help of data containing these 6 variables
# See Notebooks/lime.ipynb for more info

columns = ['DER_mass_MMC', 'DER_mass_vis', 'DER_mass_transverse_met_lep', 'PRI_tau_pt', 'PRI_met_sumet', 'DER_mass_jet_jet', 'Weight', 'Label']
df = df[columns]
X, y = preprocess(df)
infer_model = fit_score(X, y) 
save(infer_model, 'models/Base_infer_XGB.pkl')


