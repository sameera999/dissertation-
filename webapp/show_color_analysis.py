import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show_color_analysis(df):
    st.title("Color Analysis")

    color = st.selectbox('Select color', df['color'].unique())
    
    # Filter the dataframe for the selected color
    df_color = df[df['color'] == color]
    
    # Sum sales across all weeks
    df_color['total_sales'] = df_color.loc[:, 'w1_sales':'w12_sales'].sum(axis=1)
    
    # For each feature in the list
    for feature in ['fabric', 'category', 'season']:
        st.subheader(f"{feature.capitalize()} Analysis for selected color")
        
        # Group by feature and sum total_sales
        feature_sales_in_color = df_color.groupby(feature)['total_sales'].sum()
        
        # Create two columns for the chart and the summary
        col1, col2 = st.columns(2)

        # Display bar chart on the left column
        col1.bar_chart(feature_sales_in_color)

        # Display custom summary on the right column
        col2.subheader("Summary:")
        highest_performer = feature_sales_in_color.idxmax()
        lowest_performer = feature_sales_in_color.idxmin()
        highest_sales = feature_sales_in_color.max()
        lowest_sales = feature_sales_in_color.min()
        total_sales = feature_sales_in_color.sum()
        col2.write(f"{feature.capitalize()} with highest sales: {highest_performer} (Sales: {highest_sales})")
        col2.write(f"{feature.capitalize()} with lowest sales: {lowest_performer} (Sales: {lowest_sales})")
        col2.write(f"Total {feature} sales for selected color: {total_sales}")