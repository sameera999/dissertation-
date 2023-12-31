import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from load_data import load_data
from show_overview import show_overview
from show_sales import show_sales
from show_season import show_season
from show_color_analysis import show_color_analysis
from show_category_analysis import show_category_analysis
from show_fabric_analysis import show_fabric_analysis
from show_googleTrend import show_googleTrend
from show_analysis import show_analysis

st.set_page_config(layout="wide")  # Set page to wide layout

def main():
    df = load_data()
    gt = pd.read_csv("C:/Users/Sameera/OneDrive - York St John University/MYPROJECT/dataset/vis2_gtrends_data.csv")
    # Convert 'date' column to datetime format
    gt['date'] = pd.to_datetime(gt['date'])
    # Set 'date' column as index
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dataset Overview", "Sales Analysis", "Seasonal Analysis",
                                      "Color Analysis","Category Analysis","Fabric Analysis", "Google Trends", "Model Analysis"])

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
    elif page == "Google Trends":
        show_googleTrend(gt, df)
    elif page == "Model Analysis":
        show_analysis()
   

if __name__ == "__main__":
    main()



    
