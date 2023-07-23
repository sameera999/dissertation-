import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show_sales(df):
    st.title("Fashion Data Analysis: Sales Analysis")
    week = st.slider('Select sales week to view', min_value=1, max_value=12, value=1, step=1)
    df_week_sales = df[f'w{week}_sales']
    fig, ax = plt.subplots(figsize=(12, 5))  # Adjust the plot size
    df_week_sales.plot(kind='line', ax=ax)
    ax.set_xlabel("Clothing ID")
    ax.set_ylabel(f"Week {week} Sales")  # Set y-axis label
    st.pyplot(fig)
    
    st.title("Cumulative Sales Analysis")
    weeks = st.slider('Select sales weeks to view', min_value=1, max_value=12, value=(1, 12), step=1)
    df_week_sales = df.loc[:, [f'w{i}_sales' for i in range(weeks[0], weeks[1]+1)]]
    df_week_sales_cumulative = df_week_sales.cumsum(axis=1)
    fig, ax = plt.subplots(figsize=(12, 5))
    df_week_sales_cumulative.mean().plot(kind='line', ax=ax)
    ax.set_xlabel("Week")
    ax.set_ylabel("Cumulative Sales")
    st.pyplot(fig)    
    
    
    st.title("Sales Boxplot")
    weeks = st.multiselect('Select sales weeks to view', options=[i for i in range(1, 13)], default=[i for i in range(1, 13)])
    df_week_sales = df.loc[:, [f'w{i}_sales' for i in weeks]]
    fig, ax = plt.subplots(figsize=(12, 5))
    df_week_sales.boxplot(ax=ax)
    ax.set_xlabel("Week")
    ax.set_ylabel("Sales")
    st.pyplot(fig)
    
    st.title("Sales Histogram")
    week = st.slider('Select sales week to views', min_value=1, max_value=12, value=1, step=1)
    df_week_sales = df[f'w{week}_sales']
    fig, ax = plt.subplots(figsize=(12, 5))
    df_week_sales.plot(kind='hist', bins=30, rwidth=0.8, ax=ax)
    ax.set_xlabel(f"Week {week} Sales")
    ax.set_ylabel("Count")
    st.pyplot(fig)