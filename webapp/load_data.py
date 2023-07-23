import pandas as pd

def load_data():
    df = pd.read_csv("C:/Users/Sameera/OneDrive - York St John University/MYPROJECT/processedData/combinedSales.csv")
    df['release_date'] = pd.to_datetime(df['release_date'])
    return df
