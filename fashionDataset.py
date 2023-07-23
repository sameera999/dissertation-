import pandas as pd
from tqdm import tqdm  # Import tqdm for progress tracking
import warnings
warnings.filterwarnings('ignore')

class FashionDataSet:    
    # Function to frame the time series data
    def frame_series(self,train_window=3, forecast_horizon=1, method='nor', sample_size=0.01):
        df = pd.read_csv("C:/Users/Sameera/OneDrive - York St John University/MYPROJECT/processedData/processedSales.csv")
        df = df.sort_values(by="release_date")
        n_samples = int(len(df) * sample_size)        
        df = df[:n_samples]
        
        X, y = [], []
       
        # List of column names for sales and discounts
        sales_columns = [f'w{i}_sales' for i in range(1, 13)]
        discount_columns = [f'w{i}_discount' for i in range(1, 13)]
        
        for (_, row) in tqdm(df.iterrows(), total=len(df)):
            sales = row[sales_columns].values
            discount = row[discount_columns].values
            
            for j in range(len(sales) - train_window - forecast_horizon + 1):
                features = row[['external_code', 'retail','season','category','color','fabric']].tolist()
                features.extend(list(sales[j : j + train_window]))  
                if method == 'exo':
                    features.extend(row[['price','category_pct_change'
                                ,'color_pct_change','fabric_pct_change']].tolist())                
                    features.extend(list(discount[j : j + train_window]))
                if method == 'fee':
                    features.extend(row[['price','category_pct_change'
                                ,'color_pct_change','fabric_pct_change']].tolist())                
                    features.extend(list(discount[j : j + train_window]))
                target = sales[j + train_window : j + train_window + forecast_horizon]
                X.append(features)
                y.append(target)
        if method == 'nor':
            print(f"-----------------------------Perfomance with normal data[sample size {sample_size*100}%, train weeks {train_window}, forecast weeks {forecast_horizon}]--------------------------------------------")
        if method == 'exo':
            print(f"-----------------------------Perfomance with exogenous data[sample size {sample_size*100}%, train weeks {train_window}, forecast weeks {forecast_horizon}]--------------------------------------------")
        if method == 'fee':
            print(f"-----------------------------Perfomance with feature engineering data[sample size {sample_size*100}%, train weeks {train_window}, forecast weeks {forecast_horizon}]--------------------------------------------")
        return X, y


   
