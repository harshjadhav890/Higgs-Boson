import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('data/training.zip')


def add_spines(ax, colour = '#2d6383', linewidth = 2, heatmap=False):
    """
    Add beautiful spines to you plots
    """
    ax.spines[['bottom', 'left', 'top', 'right']].set_visible(True)
    if heatmap==True:
        ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left', 'top', 'right']].set_color(colour)
    ax.spines[['bottom', 'left', 'top', 'right']].set_linewidth(linewidth)

def create_seaborn_plots():
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    segment = df.replace(-999, np.nan).groupby(['PRI_jet_num', 'Label']).EventId.count().reset_index()
    sns.barplot(data=segment, x='PRI_jet_num', y='EventId', hue='Label', palette = {"s": "#63d8d8", "b": "#f7f7e1"}, ax=ax1)
    add_spines(ax1, linewidth=2)
    # plt.xlabel('PRI_jet_num', fontsize= 18).set_color('#425169')
    ax1.set_xlabel('PRI_jet_num', fontsize=18, color='#425169') 
    ax1.set_ylabel('Count', fontsize=18, color='#425169') 

    segment = df.replace(-999, np.nan).groupby(['Label', 'PRI_jet_num']).EventId.count().reset_index()
    sns.barplot(data=segment, x='Label', y='EventId', hue='PRI_jet_num', palette = 'flare', ax=ax2)
    add_spines(ax2, linewidth=2)
    ax2.set_xlabel('Label', fontsize=18, color='#425169') 
    ax2.set_ylabel('Count', fontsize=18, color='#425169') 
    st.pyplot(fig)

st.set_page_config(page_title="Analysis of Number of Jets", page_icon=":rocket:", layout="wide")
st.title("🚀 Analysis of Number of Jets")
st.write("You can see that the chances of an event being a background event reduce as the number of jets increase")
create_seaborn_plots()