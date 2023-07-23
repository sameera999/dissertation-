import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show_overview(df):
    st.title("Fashion Data Analysis: Dataset Overview")
    default_columns = ['external_code', 'season','category','fabric','color']  # replace with your default columns
    columns = st.multiselect('Select columns to display', df.columns, default=default_columns)
    num_rows = st.slider('Select number of rows to view', min_value=5, max_value=100, value=5, step=5)
    st.dataframe(df.loc[:, columns].head(num_rows))    
    st.write(df[columns].describe())
    
    st.title(f"Count Plots")
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()  # Flatten to loop easily
    features = ["color", "fabric", "category", "season"]
    for ax, feature in zip(axs, features):
        feature_unique_values = df[feature].unique()
        palette = sns.color_palette("hls", len(feature_unique_values))
        sns.countplot(data=df, x=feature, ax=ax, palette=palette)
        ax.set_title(feature)
        if feature in ['fabric', 'category']:
            ax.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    st.pyplot(fig)