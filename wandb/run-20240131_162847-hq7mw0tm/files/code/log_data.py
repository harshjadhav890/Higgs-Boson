import wandb
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import os
from dotenv import load_dotenv
os.getcwd()

## Preprocessing ------------------------------------------------------------------------#
df = pd.read_csv("data/training.zip", low_memory=False)

def preprocess(df):
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df['Label'])
    parquet_filename = 'data/preprocessed_data.parquet'
    df.to_parquet(parquet_filename, engine='fastparquet')
    X = df.drop(['Label', 'Weight'], axis=1)
    y = df['Label']
    return X, y

X, y = preprocess(df)


## Using wandb to log data --------------------------------------------------------------#

## uncomment when running locally. left out to be used by github workflows
load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# Logging into wandb
wandb.login(key=WANDB_API_KEY)

# 1. Starting a new run
run = wandb.init(project="Higgs-Boson", job_type = 'create_data')

# 2. Creating a data artifact
df = wandb.Artifact(
    name='df_preprocessed',
    type='dataset'
    )
df.add_file(local_path='data/preprocessed_data.parquet')
run.log_artifact(df)

# 3. logging the artifact and finishing the run
# run.log_artifact(df_large)
run.finish()