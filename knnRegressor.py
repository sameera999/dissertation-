import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from fashionDataset import FashionDataSet; 
from metrics import ErrorMetrics
import argparse
import warnings
warnings.filterwarnings('ignore')

def runKNN(args):  
    fd = FashionDataSet()
    er = ErrorMetrics()

    X, y = fd.frame_series(args.train_window,args.forecast_horizon,args.method, args.sample_size)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create and train the KNN model
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)

    er.calculate_errors("KNN", y_test, y_pred)

parser = argparse.ArgumentParser()
parser.add_argument("--sample_size", type=int, default=1)
parser.add_argument("--train_window", type=int, default=3)
parser.add_argument("--forecast_horizon", type=int, default=6)
parser.add_argument("--method", type=str, default="fee")

args = parser.parse_args()
runKNN(args)
