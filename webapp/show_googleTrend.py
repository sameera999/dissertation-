import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Define lists for each category
categories = ['long sleeve', 'culottes', 'miniskirt', 'short sleeves', 'printed shirt', 'short cardigan', 'solid color top', 'trapeze dress', 'sleeveless', 'long cardigan', 'sheath dress', 'short coat', 'medium coat', 'doll dress', 'long dress', 'shorts', 'long coat', 'jumpsuit', 'drop sleeve', 'patterned top', 'kimono dress', 'medium cardigan', 'shirt dress', 'maxi', 'capris', 'gitana skirt', 'long duster']
colors = ['yellow', 'brown', 'blue', 'grey', 'green', 'black', 'red', 'white', 'orange', 'violet']
fabrics = ['acrylic', 'scuba crepe', 'tulle', 'angora', 'faux leather', 'georgette', 'lurex', 'nice', 'crepe', 'satin cotton', 'silky satin', 'fur', 'matte jersey', 'plisse', 'velvet', 'lace', 'cotton', 'piquet', 'plush', 'bengaline', 'jacquard', 'frise', 'technical', 'cady', 'dark jeans', 'light jeans', 'ity', 'plumetis', 'polyviscous', 'dainetto', 'webbing', 'foam rubber', 'chanel', 'marocain', 'macrame', 'embossed', 'heavy jeans', 'nylon', 'tencel', 'paillettes', 'chambree', 'chine crepe', 'muslin cotton or silk', 'linen', 'tactel', 'viscose twill', 'cloth', 'mohair', 'mutton', 'scottish', 'milano stitch', 'devore', 'hron', 'ottoman', 'fluid', 'flamed', 'fluid polyviscous', 'shiny jersey', 'goose']

def show_googleTrend(gt, df):   
    
    st.title("Google Trends Data Analysis")

    # Create a list of columns to select from, remove 'date' from the list
    selected_categories = st.multiselect('Select categories to display', categories)
    selected_colors = st.multiselect('Select colors to display', colors)
    selected_fabrics = st.multiselect('Select fabrics to display', fabrics)

    selected_columns = selected_categories + selected_colors + selected_fabrics

    if not selected_columns:
        st.info("Please select at least one column for the analysis.")
        return

    # Calculate number of rows needed for plots
    num_plots = len(selected_columns)
    num_rows = num_plots // 2
    num_rows += num_plots % 2

    fig = plt.figure(figsize=(14, num_rows*7))

    # Loop over selected columns to create subplot for each
    for idx, selected_column in enumerate(selected_columns, start=1):
        ax = fig.add_subplot(num_rows, 2, idx)
        gt[selected_column].plot(ax=ax)
        ax.set_xlabel('Date')
        ax.set_ylabel('Trend Value')
        ax.set_title(f'Google Trends Over Time for {selected_column}')

    plt.tight_layout()
    st.pyplot(fig)
    
    gtrendwith_sales(gt, df)
    
    

def gtrendwith_sales(gt, df):
    st.title("Google trends against the slaes")
    df.set_index('release_date', inplace=True)
    gt.set_index('date', inplace=True)
    # Group by release_date and calculate the sum of sales
    sales_df = df[['w1_sales', 'w2_sales', 'w3_sales', 'w4_sales', 'w5_sales', 'w6_sales', 'w7_sales', 'w8_sales', 'w9_sales', 'w10_sales', 'w11_sales', 'w12_sales']].resample('M').sum()

    fig, axs = plt.subplots(3, 1, figsize=(14, 20))

    # Plot total sales over time
    sales_df.sum(axis=1).plot(ax=axs[0], marker='o')
    axs[0].set_title('Total Sales Over Time')
    axs[0].set_ylabel('Sales')

    # Plot number of products released over time
    df.resample('M').size().plot(ax=axs[1], marker='o')
    axs[1].set_title('Number of Products Released Over Time')
    axs[1].set_ylabel('Number of Products')
      
    # Plot Google Trends data for each fabric, color, and category
    gt.plot(ax=axs[2])
    axs[2].set_title('Google Trends Data Over Time')

    plt.tight_layout()
    st.pyplot(fig)