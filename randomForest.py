import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm  # Import tqdm for progress tracking
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from fashionDataset import FashionDataSet; 
from metrics import ErrorMetrics
import argparse
import warnings
warnings.filterwarnings('ignore')

def runRandomForest(args):  
    fd = FashionDataSet()
    er = ErrorMetrics()

    X, y = fd.frame_series(args.train_window,args.forecast_horizon,args.method, args.sample_size)
    # Combine X and y using numpy hstack
    # pca = PCA(0.95)
    # x_pca = pca.fit_transform(X)
    # X = x_pca

    # get X, and y from fashion dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create and train the Random Forest Regressor model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    er.calculate_errors("Random Forest",y_test,y_pred)

parser = argparse.ArgumentParser()
parser.add_argument("--sample_size", type=int, default=1)
parser.add_argument("--train_window", type=int, default=3)
parser.add_argument("--forecast_horizon", type=int, default=6)
parser.add_argument("--method", type=str, default="")

args = parser.parse_args()
runRandomForest(args)
