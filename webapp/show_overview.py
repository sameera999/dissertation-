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
    
    st.title(f"Count Plots and Sales Summary")
    # Sum sales across all weeks
    df['total_sales'] = df.loc[:, 'w1_sales':'w12_sales'].sum(axis=1)

    features = ["color", "fabric", "category", "season"]
    for feature in features:
        feature_unique_values = df[feature].unique()
        palette = sns.color_palette("hls", len(feature_unique_values))
        
        # Create two columns for the chart and the summary
        col1, col2 = st.columns(2)

        # Count Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.countplot(data=df, x=feature, ax=ax, palette=palette)
        ax.set_title(f'Count Plot for {feature}')
        if feature in ['fabric', 'category']:
            ax.tick_params(axis='x', rotation=90)
        col1.pyplot(fig)
        
        # Sales Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        df.groupby(feature)['total_sales'].sum().plot(kind='bar', ax=ax, color=palette)
        ax.set_title(f'Sales Plot for {feature}')
        ax.tick_params(axis='x', rotation=90)
        col2.pyplot(fig)

        # Calculate total sales for each unique value of the feature
        feature_sales = df.groupby(feature)['total_sales'].sum()

        # Find the best and worst performers
        best_performer = feature_sales.idxmax()
        worst_performer = feature_sales.idxmin()

        # Mean sales
        mean_sales = df['total_sales'].mean()

        # Total unique items
        total_unique = df[feature].nunique()

        # Percentage contribution to total sales
        total_sales = df['total_sales'].sum()
        best_sales_contribution = feature_sales[best_performer] / total_sales
        worst_sales_contribution = feature_sales[worst_performer] / total_sales

        # Information for the feature
        feature_info = f"""
        - **Best performer:** {best_performer} with total sales of {feature_sales[best_performer]}
        - **Worst performer:** {worst_performer} with total sales of {feature_sales[worst_performer]}
        - **Mean sales:** {mean_sales}
        - **Total number of unique {feature}:** {total_unique}
        - **Sales contribution of best performer:** {best_sales_contribution * 100:.2f}%
        - **Sales contribution of worst performer:** {worst_sales_contribution * 100:.2f}%
        - **Predictive Insights for {feature}:** If the majority of items in the future belong to the most common {feature} ({df[feature].mode()[0]}), expect a higher count of items, which may also lead to increased sales. If a future item has the {feature} of the best sales performer ({best_performer}), it may have higher chances of obtaining high sales.
        - **Analytical Findings for {feature}:** The wide disparity in sales between the best and worst performers in the {feature} attribute suggests significant variability in customer preference based on {feature}. The significant difference in count between the most and least common {feature} could indicate market saturation or lack of variety in certain {feature}.
        """
        col2.markdown(feature_info)
