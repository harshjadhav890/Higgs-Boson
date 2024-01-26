import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier

# Load the saved model from the file
with open('models/Baseline_XGB.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

