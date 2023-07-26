import pandas as pd
from tqdm import tqdm  # Import tqdm for progress tracking
import warnings
import numpy as np
import torch
from torch.utils.data import TensorDataset
warnings.filterwarnings('ignore')

class FashionDataSet:    
    # Function to frame the time series data
    def frame_series(self,train_window=3, forecast_horizon=1, method='nor', sample_size=1):
        df = pd.read_csv("C:/Users/Sameera/OneDrive - York St John University/MYPROJECT/processedData/processedSales.csv")
        df = df.sort_values(by="release_date")
        n_samples = int(len(df) * sample_size)        
        df = df[:n_samples]
        
        X, y = [], []
       
        # List of column names for sales and discounts
        sales_columns = [f'w{i}_sales' for i in range(1, 13)]
        discount_columns = [f'w{i}_discount' for i in range(1, 13)]        
        
        
        for (_, row) in tqdm(df.iterrows(), total=len(df),desc="Preparing Dataset"):           
            sales = row[sales_columns].values.astype(float)  # Convert sales to float
            discount = row[discount_columns].values
            
            for j in range(len(sales) - train_window - forecast_horizon + 1):
                features = row[['external_code','season','category','color','fabric']].tolist()
                sales_window = sales[j : j + train_window]
                discount_window = discount[j : j + train_window]
                features.extend(list(sales_window))  
                if method == 'exo':
                    features.extend(row[['release_year','releae_quarter','releae_month','releae_week','is_weekend','price','category_pct_change'
                                ,'color_pct_change','fabric_pct_change']].tolist())                
                    features.extend(list(discount_window))
                if method == 'fee':
                    # avg category sales 
                    avg_category_sales = self.calculate_avg_category_sales(df, row['category'], (j+1), (j + 1 + train_window))
                    features.append(avg_category_sales)

                    avg_fabric_sales = self.calculate_avg_fabric_sales(df, row['fabric'], (j+1), (j + 1 + train_window))
                    features.append(avg_fabric_sales)

                    avg_color_sales = self.calculate_avg_color_sales(df, row['color'], (j+1), (j + 1 + train_window))
                    features.append(avg_color_sales)
                    
                    sales_rate_of_change, discount_rate_of_change = self.calculate_rate_of_change(sales, discount, (j+1), (j + 1 + train_window))
                    features.extend([sales_rate_of_change, discount_rate_of_change])
                                        
                    rolling_mean_sales, rolling_std_sales, rolling_mean_discount, rolling_std_discount = self.calculate_rolling_statistics(sales, discount, j + 1, j + train_window)
                    features.extend([rolling_mean_sales, rolling_std_sales, rolling_mean_discount, rolling_std_discount])                            

                    
                target = sales[j + train_window : j + train_window + forecast_horizon]
                X.append(features)
                y.append(target)
                
        if method == 'nor':
            print(f"-----------------------------Perfomance with normal data[sample size {sample_size*100}%, train weeks {train_window}, forecast weeks {forecast_horizon}]--------------------------------------------")
        if method == 'exo':
            print(f"-----------------------------Perfomance with exogenous data[sample size {sample_size*100}%, train weeks {train_window}, forecast weeks {forecast_horizon}]--------------------------------------------")
        if method == 'fee':
            print(f"-----------------------------Perfomance with feature engineering data[sample size {sample_size*100}%, train weeks {train_window}, forecast weeks {forecast_horizon}]--------------------------------------------")
        return np.array(X), np.array(y)   
    

    def calculate_trend(self, series):
        # Add your trend calculation here
        trend = np.polyfit(range(len(series)), series, 1)[0]
        return trend
    
    def calculate_avg_category_sales(self, df, category, start_week, end_week):
        # Filter dataframe for the given category and weeks
        category_df = df[df['category'] == category]
        
        # Calculate the average sales in the specified weeks
        avg_sales = category_df[[f'w{i}_sales' for i in range(start_week, end_week + 1)]].mean().mean()
        
        return avg_sales

    def calculate_avg_fabric_sales(self, df, fabric, start_week, end_week):
        # Filter dataframe for the given fabric
        fabric_df = df[df['fabric'] == fabric]
        
        # Calculate the average sales in the specified weeks
        avg_sales = fabric_df[[f'w{i}_sales' for i in range(start_week, end_week + 1)]].mean().mean()
        
        return avg_sales

    def calculate_avg_color_sales(self, df, color, start_week, end_week):
        # Filter dataframe for the given color
        color_df = df[df['color'] == color]
        
        # Calculate the average sales in the specified weeks
        avg_sales = color_df[[f'w{i}_sales' for i in range(start_week, end_week + 1)]].mean().mean()
        
        return avg_sales

    def calculate_rate_of_change(self, sales, discount, start_week, end_week):
        # Calculate the rate of change in sales and discount between the start and end week
        sales_rate_of_change = (sales[end_week - 1] - sales[start_week - 1]) / sales[start_week - 1] if sales[start_week - 1] != 0 else 0
        discount_rate_of_change = (discount[end_week - 1] - discount[start_week - 1]) / discount[start_week - 1] if discount[start_week - 1] != 0 else 0
        
        return sales_rate_of_change, discount_rate_of_change
    
       
    # Calculate the rolling mean and standard deviation of the sales and discounts between the start and end week
    def calculate_rolling_statistics(self, sales, discount, start_week, end_week):
        # Calculate the rolling mean and standard deviation for the weeks between `start_week` and `end_week`
        rolling_mean_sales = np.mean(sales[start_week - 1 : end_week])
        rolling_std_sales = np.std(sales[start_week - 1 : end_week])
        rolling_mean_discount = np.mean(discount[start_week - 1 : end_week])
        rolling_std_discount = np.std(discount[start_week - 1 : end_week])

        return rolling_mean_sales, rolling_std_sales, rolling_mean_discount, rolling_std_discount