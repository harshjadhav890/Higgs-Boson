# This file must run automatically whenever there are any changes to the data
# change in data -> run wandb_log_data.py -> run wandb_log_model.py

import wandb
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from modelling import preprocess, fit_score, save

import os
from dotenv import load_dotenv

## uncomment when running locally. left out to be used by github workflows
load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# Logging into wandb
wandb.login(key=WANDB_API_KEY)

# import fastparquet
parquet_file_path = 'data/preprocessed_data.parquet'
# Read the Parquet file into a Pandas DataFrame using fastparquet
df = pd.read_parquet(parquet_file_path, engine='fastparquet')

# setup parameters for xgboost
params = {
    "eta" : 0.1,
    "max_depth": 6,
    "nthread" : 4,
}

# Train and save the model locally
local_model_path = 'models/Tuned_model.pkl'
X, y = preprocess(df)
model, metrics = fit_score(X, y, **params)
save(model, local_model_path)

def log_model(model_name, local_model_path, wandb_data_path = 'harsh-ajay-jadhav/Higgs-Boson/df_preprocessed:v0', metrics=None):
    run = wandb.init(
        project="Higgs-Boson",
        config=params,
        job_type = 'train_model'
    )

    run.use_artifact(wandb_data_path, type='dataset')

    model_artifact = wandb.Artifact(
            name=model_name,
            type='model'
        )

    model_artifact.add_file(local_path=local_model_path)
    run.log_artifact(model_artifact)
    if metrics is not None:
        run.log(metrics)

    run.finish()

log_model('tuned_model', local_model_path, metrics=metrics)
