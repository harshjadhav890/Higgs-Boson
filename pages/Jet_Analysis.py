import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('Higgs-Boson/data/training.zip')


def add_spines(ax, colour = '#2d6383', linewidth = 2, heatmap=False):
    """
    Add beautiful spines to you plots
    """
    ax.spines[['bottom', 'left', 'top', 'right']].set_visible(True)
    if heatmap==True:
        ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left', 'top', 'right']].set_color(colour)
    ax.spines[['bottom', 'left', 'top', 'right']].set_linewidth(linewidth)

st.set_page_config(page_title="Analysis of Number of Jets", page_icon=":rocket:", layout="wide")

# Chart I ----------------------------------------------------------------------------------------------------------------#

st.subheader("🚀 Analysis of Number of Jets")
st.write("You can see that the chances of an event being a background event reduce as the number of jets increase")

## Code to actually plot the chart and display through streamlit
# sns.set_theme(style="whitegrid")
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
# segment = df.replace(-999, np.nan).groupby(['PRI_jet_num', 'Label']).EventId.count().reset_index()
# sns.barplot(data=segment, x='PRI_jet_num', y='EventId', hue='Label', palette = {"s": "#63d8d8", "b": "#f7f7e1"}, ax=ax1)
# add_spines(ax1, linewidth=2)
# ax1.set_xlabel('PRI_jet_num', fontsize=18, color='#425169') 
# ax1.set_ylabel('Count', fontsize=18, color='#425169') 

# segment = df.replace(-999, np.nan).groupby(['Label', 'PRI_jet_num']).EventId.count().reset_index()
# sns.barplot(data=segment, x='Label', y='EventId', hue='PRI_jet_num', palette = 'flare', ax=ax2)
# add_spines(ax2, linewidth=2)
# ax2.set_xlabel('Label', fontsize=18, color='#425169') 
# ax2.set_ylabel('Count', fontsize=18, color='#425169') 
# st.pyplot(fig)

## We'll take a static approach instead and load the plot image directly. This will save loading time
image1 = Image.open("Higgs-Boson/images/jet_bar_chart.png")
st.image(image1, caption='Your Image', use_column_width=True)


# Chart II ----------------------------------------------------------------------------------------------------------------#

st.subheader("🚀 Distribution wrt Number of Jets")
st.write("Some of the variables show an increasing trend as the number of jets increase")

## Code to actually plot the chart and display through streamlit
# def linear_to_2d(index):
#     row_index = index // 3
#     col_index = index % 3
#     return (row_index, col_index)

# sns.set_theme(style="white")
# def violin(ax, n, y, top, bottom):
#     # plt.subplot(2, 3, n+1)
#     ax1 = ax[linear_to_2d(n)]
#     sns.violinplot(data=df.replace(-999, np.nan), x='PRI_jet_num', y=y, hue="Label",
#                    split=True, inner="quart", palette={"s": "#63d8d8", "b": "#f7f7e1"},ax=ax1)
#     add_spines(ax=ax1, colour= '#275773', linewidth=1)
#     if top is not None:  
#         ax1.set_ylim(bottom, top)

# plt.figure(figsize=(20, 9))
# columns_selected = [("DER_pt_h", 500, -50),("DER_pt_tot", 250, -10),("DER_sum_pt", 1000, 0),("PRI_met_sumet", 1000, 0),("PRI_jet_leading_pt", 450, 0),("PRI_jet_all_pt", 750, 0)]
# fig, ax = plt.subplots(2, 3, figsize=(20, 7))
# for n, y in enumerate(columns_selected):
#     violin(ax, n, y[0], y[1], y[2])
# st.pyplot(fig)

## We'll take a static approach instead and load the plot image directly. This will save loading time
image2 = Image.open("Higgs-Boson/images/jet_distribution_violin_chart.png")
st.image(image2, caption='Your Image', use_column_width=True)

