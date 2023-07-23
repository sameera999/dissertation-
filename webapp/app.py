import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")  # Set page to wide layout

def load_data():
    df = pd.read_csv("C:/Users/Sameera/OneDrive - York St John University/MYPROJECT/processedData/combinedSales.csv")
    df['release_date'] = pd.to_datetime(df['release_date'])
    return df

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

        # Display bar chart on the left column
        col1.bar_chart(feature_sales_in_season)

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


def show_release_date(df):
    st.title("Fashion Data Analysis: Release Date Analysis")
    year = st.slider('Select release year', min_value=df['release_year'].min(), max_value=df['release_year'].max(), value=df['release_year'].min())
    df_year = df[df['release_year'] == year]
    fig, ax = plt.subplots(figsize=(12, 8))  # Adjust the plot size
    df_year['release_year'].value_counts().sort_index().plot(kind='bar', ax=ax)
    st.pyplot(fig)

df = load_data()
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dataset Overview", "Sales Analysis", "Seasonal Analysis",
                                  "Color Analysis","Category Analysis","Fabric Analysis", "Release Date Analysis"])

if page == "Dataset Overview":
    show_overview(df)
elif page == "Sales Analysis":
    show_sales(df)
elif page == "Seasonal Analysis":
    show_season(df)
elif page == "Color Analysis":
    show_color_analysis(df)
elif page == "Category Analysis":
    show_category_analysis(df)
elif page == "Fabric Analysis":
    show_fabric_analysis(df)
elif page == "Release Date Analysis":
    show_release_date(df)
