# The Main page of our streamlit app

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# Load the saved model from the file
with open('models/Base_infer_XGB.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

st.set_page_config(page_title="⚛️Searching for God particle")
st.title("⚛️ Searching for God particle")
st.write("A simple web app that predicts whether the given event is a signal or a background event")

def command():
    print("")

with st.form('input_form'):

    # Create two columns
    col1, col2 = st.columns(2)

    # Sliders in the first column
    with col1:
        PRI_tau_pt = st.slider("PRI_tau_pt: ", 20, 765, value=20, format="%d")
        DER_mass_MMC = st.slider("DER_mass_MMC: ", -999, 1200, value=140, format="%d")
        DER_mass_vis = st.slider("DER_mass_vis: ", 0, 1350, value=0, format="%d")

    # Sliders in the second column
    with col2:
        PRI_met_sumet = st.slider("PRI_met_sumet: ", 13, 2004, value=13, format="%d")
        DER_mass_transverse_met_lep = st.slider("DER_mass_transverse_met_lep: ", 0, 691, value=0, format="%d")
        DER_mass_jet_jet = st.slider("DER_mass_jet_jet: ", -999, 4975, value=2000, format="%d")
    
    submit = st.form_submit_button("Predict")

if submit:
    data = np.array([PRI_tau_pt,PRI_met_sumet,DER_mass_MMC,DER_mass_vis,DER_mass_transverse_met_lep, 
                        DER_mass_jet_jet]).reshape(1,-1)
    # data = np.array([20,13,999,1350,0, 999]).reshape(1,-1)
    pred = model.predict(data)
    
    if pred[0]:
        # st.write("This is a Signal event")
        st.success("This is a Signal event")
    else:
        st.error("This is a Background event")


