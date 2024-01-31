import wandb
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import os
from dotenv import load_dotenv

## Preprocessing ------------------------------------------------------------------------#

# def preprocess(df):
#     label_encoder = LabelEncoder()
#     df['Label'] = label_encoder.fit_transform(df['Label'])
#     parquet_filename = 'data/preprocessed_data.parquet'
#     df.to_parquet(parquet_filename, engine='fastparquet')
#     X = df.drop(['Label', 'Weight'], axis=1)
#     y = df['Label']
#     return X, y

# X, y = preprocess(df)


## Using wandb to log data --------------------------------------------------------------#

def log_data(data_name, local_data_path):
    # 1. Starting a new run
    run = wandb.init(project="Higgs-Boson", job_type = 'create_data')

    # 2. Creating a data artifact
    df = wandb.Artifact(
        name=data_name,
        type='dataset'
        )
    df.add_file(local_path=local_data_path)
    run.log_artifact(df)

    run.finish()

# Fetching the API_KEY through env variables
load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# Logging into wandb
wandb.login(key=WANDB_API_KEY)

log_data(data_name='df_preprocessed', local_data_path = 'data/preprocessed_data.parquet')