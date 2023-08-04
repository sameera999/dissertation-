import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

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
    
    st.subheader("Sales Over Time")
    # Sum sales across all weeks
    df['total_sales'] = df.loc[:, 'w1_sales':'w12_sales'].sum(axis=1)
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 5)) 
    ax.scatter(df['release_date'], df['total_sales'])
    ax.set_xlabel('Release Date')
    ax.set_ylabel('Total Sales')
    ax.set_title('Sales Over Time')    
    st.pyplot(fig)
        
    st.subheader("How discount affect to sales perfoamnce")
    fig, ax = plt.subplots(figsize=(12, 6))
    # Iterate through all weeks
    for i in range(1, 13):
        sales_column = f'w{i}_sales'
        discount_column = f'w{i}_discount'
        ax.scatter(df[discount_column], df[sales_column], alpha=0.5)

    ax.set_title('Sales vs. Discount for all weeks')
    ax.set_xlabel('Discount')
    ax.set_ylabel('Sales')
    st.pyplot(fig)
    
    st.subheader("How discount affect to best perfoamed products")
    df['total_sales'] = df.loc[:, 'w1_sales':'w12_sales'].sum(axis=1)
    top_products = df.nlargest(5, 'total_sales')['external_code']

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))  # We changed this line
    axs = axs.flatten()  # Flatten to loop easily

    # Remove the last (unused) subplot
    fig.delaxes(axs[-1])

    for product, ax in zip(top_products, axs):
        product_data = df[df['external_code'] == product]
        
        ax.plot(range(1, 13), product_data.loc[:, 'w1_sales':'w12_sales'].values[0], label='Sales')
        ax.plot(range(1, 13), product_data.loc[:, 'w1_discount':'w12_discount'].values[0]*1000, label='Discount (%)')  # Multiplying discount by 100
        
        ax.set_title(f'Product {product}')
        ax.set_xticks(range(1, 13))
        ax.legend()

    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("How discount affect to worst perfoamed products")
    df['total_sales'] = df.loc[:, 'w1_sales':'w12_sales'].sum(axis=1)
    worst_products = df.nsmallest(5, 'total_sales')['external_code']

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))  # We changed this line
    axs = axs.flatten()  # Flatten to loop easily

    # Remove the last (unused) subplot
    fig.delaxes(axs[-1])

    for product, ax in zip(worst_products, axs):
        product_data = df[df['external_code'] == product]
        
        ax.plot(range(1, 13), product_data.loc[:, 'w1_sales':'w12_sales'].values[0], label='Sales')
        ax.plot(range(1, 13), product_data.loc[:, 'w1_discount':'w12_discount'].values[0], label='Discount (%)')  # Multiplying discount by 100
        
        ax.set_title(f'Product {product}')
        ax.set_xticks(range(1, 13))
        ax.legend()

    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("Sales trend Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    df.groupby('release_date')['total_sales'].sum().plot(kind='line', ax=ax)
    ax.set_xlabel("Release Date")
    ax.set_ylabel("Total Sales")
    st.pyplot(fig)
    
    st.subheader("Box Plot of Sales by Category")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x="category", y="total_sales", data=df, ax=ax)
    ax.set_xlabel("Category")
    ax.set_ylabel("Total Sales")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.subheader("Correlation Heatmap")
    # Select numeric columns only for correlation
    numeric_cols = df.select_dtypes(include=[np.number])
    corr = numeric_cols.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax, cmap='coolwarm')
    st.pyplot(fig)
    
    st.subheader("Impact of Discount on Total Sales by Category in Selected Season")
    df['total_sales'] = df.loc[:, 'w1_sales':'w12_sales'].sum(axis=1)
    df['average_discount'] = df.loc[:, 'w1_discount':'w12_discount'].mean(axis=1)    
    # Group by season and category
    bubble_data = df.groupby(['season', 'category']).agg(total_sales=('total_sales', 'sum'),
                                                         average_discount=('average_discount', 'mean')).reset_index()
    
    seasons = sorted(df['season'].unique())
    selected_season = st.selectbox('Select a season', seasons)

    # Filter data based on user input
    bubble_data = bubble_data[bubble_data['season'] == selected_season]

    # Create the bubble chart
    fig, ax = plt.subplots(figsize=(15, 8))
    scatter = ax.scatter(bubble_data['category'], bubble_data['total_sales'],
                         s=bubble_data['average_discount']*2000, # Multiply by 2000 to make bubbles visible
                         alpha=0.5, edgecolors='w')

    ax.set_xlabel('Category')
    ax.set_ylabel('Total Sales')
    ax.set_title(f'Total Sales per Category for {selected_season} Season with Average Discount Size')
    ax.grid(True)

    # Add a color bar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.ax.set_ylabel('Average Discount')

    # Rotate the x labels for better visibility
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

   







