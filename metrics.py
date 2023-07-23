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
        mae = np.mean([mean_absolute_error(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])])
        return round(mae, 2)
        
    def calculate_wape(self, name, y_test, y_pred):
        wape = np.mean([np.sum(np.abs(y_test[:, i] - y_pred[:, i])) / np.sum(y_test[:, i]) for i in range(y_test.shape[1])]) * 100
        return round(wape, 2)
        
    def calculate_rmse(self, name, y_test, y_pred):
        rmse = np.mean([sqrt(mean_squared_error(y_test[:, i], y_pred[:, i])) for i in range(y_test.shape[1])])
        return round(rmse, 2)
