# This file contains three important functions: preprocess(), fit_score(), and save()
# We import these functions to other files for reusability
# Changes made here would also be reflected in other files 
# Running this file saves 2 baseline models 
# -> A full version of the model using all variables in the dataset as well as the inference model trained on 6 variables

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import train_test_split

# Changes to the preprocess funciton will cause the data to change
# This will trigger a workflow that will automatically log data and train and log models
def preprocess(df):
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df['Label'])
    X = df.drop(['Label', 'Weight'], axis=1)
    y = df['Label']
    return X, y

def fit_score(X, y, **kwargs):
    """Fit, cross-validate and print metrics

    Args:
        X (pandas.core.frame.DataFrame): Independent variables
        y (pandas.core.frame.DataFrame): Label to predict

    Returns:
        xgboost.sklearn.XGBClassifier: A baseline XGB classifier
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    xgb_model = XGBClassifier(objective='binary:logistic', random_state=42, **kwargs)
    # xgb_model.set_params(params)
    xgb_model.fit(X_train, y_train)
    
    precision_scorer = make_scorer(precision_score)
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    precision_scores = cross_val_score(xgb_model, X_train, y_train, cv=k_fold, scoring=precision_scorer)
    # for i, precision in enumerate(precision_scores):
    #     print(f'Fold {i+1}: Precision = {precision}')
    
    y_pred = xgb_model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    # print(f'\nTest data Precision: {precision}')
    
    precision_class_0 = precision_score(y_test, y_pred, labels=[0], average=None)[0]
    precision_class_1 = precision_score(y_test, y_pred, labels=[1], average=None)[0]
    
    metrics = {'Average Precision': precision_scores.mean(), 'Test data Precision': precision, 'Precision of Background event': precision_class_0,  'Precision of Signal event': precision_class_1}
    
    return xgb_model, metrics

def save(model, path):
    with open(path, 'wb') as model_file:
        pickle.dump(model, model_file)


params = {
        "eta" : 0.1,
        "max_depth": 6,
        "nthread" : 4,
    }

df = pd.read_csv('data/training.zip', low_memory=False)
X, y = preprocess(df)
model, metrics = fit_score(X, y, **params)
print(metrics)
save(model, 'models/Baseline_XGB.pkl')

#-----------------------------------------------------------------------------------------------------#

# Here we chose 6 variables that had the biggest influence on the decisions of the model.
# Created a model only with the help of data containing these 6 variables since we cant have a lot of inputs during inference.
# See Notebooks/lime.ipynb for more info on how we chose the variables.

columns = ['DER_mass_MMC', 'DER_mass_vis', 'DER_mass_transverse_met_lep', 'PRI_tau_pt', 'PRI_met_sumet', 'DER_mass_jet_jet', 'Weight', 'Label']
df = df[columns]
X, y = preprocess(df)
infer_model = fit_score(X, y)
save(infer_model, 'models/Base_infer_XGB.pkl')


