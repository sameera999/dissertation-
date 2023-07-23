import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show_fabric_analysis(df):
    st.title("Fabric Analysis")

    fabric = st.selectbox('Select fabric', df['fabric'].unique())
    
    # Filter the dataframe for the selected fabric
    df_fabric = df[df['fabric'] == fabric]
    
    # Sum sales across all weeks
    df_fabric['total_sales'] = df_fabric.loc[:, 'w1_sales':'w12_sales'].sum(axis=1)
    
    # For each feature in the list
    for feature in ['category', 'color', 'season']:
        st.subheader(f"{feature.capitalize()} Analysis for selected fabric")
        
        # Group by feature and sum total_sales
        feature_sales_in_fabric = df_fabric.groupby(feature)['total_sales'].sum()
        
        # Create two columns for the chart and the summary
        col1, col2 = st.columns(2)

        # Display bar chart on the left column
        col1.bar_chart(feature_sales_in_fabric)

        # Display custom summary on the right column
        col2.subheader("Summary:")
        highest_performer = feature_sales_in_fabric.idxmax()
        lowest_performer = feature_sales_in_fabric.idxmin()
        highest_sales = feature_sales_in_fabric.max()
        lowest_sales = feature_sales_in_fabric.min()
        total_sales = feature_sales_in_fabric.sum()
        col2.write(f"{feature.capitalize()} with highest sales: {highest_performer} (Sales: {highest_sales})")
        col2.write(f"{feature.capitalize()} with lowest sales: {lowest_performer} (Sales: {lowest_sales})")
        col2.write(f"Total {feature} sales for selected fabric: {total_sales}")