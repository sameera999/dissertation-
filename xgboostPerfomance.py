import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from fashionDataset import FashionDataSet; 
from metrics import ErrorMetrics
import argparse
import warnings
warnings.filterwarnings('ignore')

def runXGBoost(args):  
    fd = FashionDataSet()
    er = ErrorMetrics()

    X, y = fd.frame_series(args.train_window,args.forecast_horizon,args.method, args.sample_size)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create and train the XGBoost model
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7, random_state=42)
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)

    er.calculate_errors("XGBoost", y_test, y_pred)

parser = argparse.ArgumentParser()
parser.add_argument("--sample_size", type=int, default=1)
parser.add_argument("--train_window", type=int, default=2)
parser.add_argument("--forecast_horizon", type=int, default=2)
parser.add_argument("--method", type=str, default="exo")

args = parser.parse_args()
runXGBoost(args)
