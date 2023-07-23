import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show_release_date(df):
    st.title("Fashion Data Analysis: Release Date Analysis")
    year = st.slider('Select release year', min_value=df['release_year'].min(), max_value=df['release_year'].max(), value=df['release_year'].min())
    df_year = df[df['release_year'] == year]
    fig, ax = plt.subplots(figsize=(12, 8))  # Adjust the plot size
    df_year['release_year'].value_counts().sort_index().plot(kind='bar', ax=ax)
    st.pyplot(fig)