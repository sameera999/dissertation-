import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def show_season(df):
    st.title("Seasonal Analysis")

    season = st.selectbox('Select season', df['season'].unique())
    
    # Filter the dataframe for the selected season
    df_season = df[df['season'] == season]
    
    # Sum sales across all weeks
    df_season['total_sales'] = df_season.loc[:, 'w1_sales':'w12_sales'].sum(axis=1)
    
    # For each feature in the list
    for feature in ['fabric', 'color', 'category']:
        st.subheader(f"Seasonal {feature.capitalize()} Analysis")
        
        # Group by feature and sum total_sales
        feature_sales_in_season = df_season.groupby(feature)['total_sales'].sum()
        
        # Create two columns for the chart and the summary
        col1, col2 = st.columns(2)

        # Create a bar chart with matplotlib and display it on the left column
        fig, ax = plt.subplots(figsize=(10, 5))  # Adjust the figure size here
        ax.bar(feature_sales_in_season.index, feature_sales_in_season, color=sns.color_palette('hls', len(feature_sales_in_season)))
        plt.xticks(rotation=90 if feature in ['fabric', 'category'] else 0)
        col1.pyplot(fig)

        # Display custom summary on the right column
        col2.subheader("Summary:")
        highest_performer = feature_sales_in_season.idxmax()
        lowest_performer = feature_sales_in_season.idxmin()
        highest_sales = feature_sales_in_season.max()
        lowest_sales = feature_sales_in_season.min()
        total_sales = feature_sales_in_season.sum()
        col2.write(f"{feature.capitalize()} with highest sales: {highest_performer} (Sales: {highest_sales})")
        col2.write(f"{feature.capitalize()} with lowest sales: {lowest_performer} (Sales: {lowest_sales})")
        col2.write(f"Total {feature} sales in the season: {total_sales}")
