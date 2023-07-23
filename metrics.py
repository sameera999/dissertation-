from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class ErrorMetrics:
    def calculate_errors(self, name, y_test, y_pred):
        metrics = {
            "(MAE)": self.calculate_mae(name, y_test, y_pred),
            "(WAPE)": self.calculate_wape(name, y_test, y_pred),
            "(RMSE)": self.calculate_rmse(name, y_test, y_pred)
        }
        df = pd.DataFrame(metrics, index=[name])
        print(df)
    
    def calculate_mae(self, name, y_test, y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        return round(mae, 2)
        
    def calculate_wape(self, name, y_test, y_pred):
        total_abs_error = np.sum(np.abs(y_test - y_pred))
        total_actual = np.sum(y_test)
        wape = (total_abs_error / total_actual) * 100      
        return round(wape, 2)
        
    def calculate_rmse(self, name, y_test, y_pred):
        mse = mean_squared_error(y_test, y_pred)
        rmse = sqrt(mse)
        return round(rmse, 2)
    