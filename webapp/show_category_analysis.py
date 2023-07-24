import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show_category_analysis(df):
    st.title("Category Analysis")
    category = st.selectbox('Select category', df['category'].unique())    
    # Filter the dataframe for the selected category
    df_category = df[df['category'] == category]    
    # Sum sales across all weeks
    df_category['total_sales'] = df_category.loc[:, 'w1_sales':'w12_sales'].sum(axis=1)    
    # For each feature in the list
    for feature in ['fabric', 'color', 'season']:
        st.subheader(f"{feature.capitalize()} Analysis for selected category")        
        # Group by feature and sum total_sales
        feature_sales_in_category = df_category.groupby(feature)['total_sales'].sum()        
        # Create two columns for the chart and the summary
        col1, col2 = st.columns(2)
        # Display bar chart on the left column
        col1.bar_chart(feature_sales_in_category)
        # Display custom summary on the right column
        col2.subheader("Summary:")
        highest_performer = feature_sales_in_category.idxmax()
        lowest_performer = feature_sales_in_category.idxmin()
        highest_sales = feature_sales_in_category.max()
        lowest_sales = feature_sales_in_category.min()
        total_sales = feature_sales_in_category.sum()
        col2.write(f"{feature.capitalize()} with highest sales: {highest_performer} (Sales: {highest_sales})")
        col2.write(f"{feature.capitalize()} with lowest sales: {lowest_performer} (Sales: {lowest_sales})")
        col2.write(f"Total {feature} sales for selected category: {total_sales}")
        
    st.subheader("Sales Trend Comparison Between Categories")
    categories = df['category'].unique()
    fig, ax = plt.subplots(figsize=(12, 6))
    for category in categories:
        df_category = df[df['category'] == category]
        df_category.loc[:, 'w1_sales':'w12_sales'].mean().plot(label=category, ax=ax)

    ax.set_xlabel('Week')
    ax.set_ylabel('Average Sales')
    ax.legend()
    st.pyplot(fig)