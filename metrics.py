# Importing necessary libraries for metrics calculation and data manipulation
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import pandas as pd
import warnings

# Suppressing all warnings to keep the console clean
warnings.filterwarnings('ignore')

# Defining a class ErrorMetrics to compute various evaluation metrics for predictions
class ErrorMetrics:
    
    # Main method to calculate errors and print them in a formatted DataFrame
    def calculate_errors(self, name, y_test, y_pred):
        metrics = {
            "(MAE)": self.calculate_mae(name, y_test, y_pred),      # Mean Absolute Error
            "(WAPE)": self.calculate_wape(name, y_test, y_pred),    # Weighted Absolute Percentage Error
            "(RMSE)": self.calculate_rmse(name, y_test, y_pred)     # Root Mean Squared Error
        }
        # Converting metrics dictionary into DataFrame for visualization
        df = pd.DataFrame(metrics, index=[name])
        print(df)
    
    # Method to calculate Mean Absolute Error for multivariate time series data
    def calculate_mae(self, name, y_test, y_pred):
        mae = np.mean([mean_absolute_error(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])])
        return round(mae, 2)
        
    # Method to calculate Weighted Absolute Percentage Error for multivariate time series data
    def calculate_wape(self, name, y_test, y_pred):
        wape = np.mean([np.sum(np.abs(y_test[:, i] - y_pred[:, i])) / np.sum(y_test[:, i]) for i in range(y_test.shape[1])]) * 100
        return round(wape, 2)
        
    # Method to calculate Root Mean Squared Error for multivariate time series data
    def calculate_rmse(self, name, y_test, y_pred):
        rmse = np.mean([sqrt(mean_squared_error(y_test[:, i], y_pred[:, i])) for i in range(y_test.shape[1])])
        return round(rmse, 2)
