# Tune and save model to models/Tuned_model.pkl
# Saves your parameters to the model_config.json file
# Making changes to the tuned model will trigger the workflow to run wandb_log_model.py

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from modelling import preprocess, fit_score, save

# You model tuning code goes here #

# ------------------------------- #

def create_param_file():
    # Define your final model parameters as a dictionary
    params = {
    "eta" : 0.1,
    "max_depth": 6,
    "nthread" : 4
    }
    # Save the parameters to a JSON file
    with open('model_config.json', 'w') as json_file:
        json.dump(params, json_file, indent=4)
    return params

df = pd.read_csv('data/training.zip', low_memory=False)
local_model_path = 'models/Tuned_model.pkl'
X, y = preprocess(df)
params = create_param_file()
model, metrics, cm = fit_score(X, y, **params)
save(model, local_model_path)