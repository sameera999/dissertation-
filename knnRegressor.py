import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from fashionDataset import FashionDataSet; 
from metrics import ErrorMetrics
import argparse
import warnings
import joblib
warnings.filterwarnings('ignore')

def runKNN(args):  
    fd = FashionDataSet()
    er = ErrorMetrics()

    X, y = fd.frame_series(args.train_window,args.forecast_horizon,args.method, args.sample_size)

    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=0)

    # Create and train the KNN model
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)

    er.calculate_errors("KNN", y_test, y_pred)
    # Save y_test and y_pred to a CSV file
    result_df = pd.DataFrame({
        'y_test': y_test.flatten(),
        'y_pred': y_pred.flatten()
    })
    result_df.to_csv('C:/Users/Sameera/OneDrive - York St John University/MYPROJECT/processedData/KNN_model_results.csv', index=False)
    
     # Save the trained model
    joblib.dump(knn_model, 'C:/Users/Sameera/OneDrive - York St John University/MYPROJECT/models/KNN.pkl')

    # # Save the feature importance
    # feature_importances = pd.DataFrame(knn_model.feature_importances_, index = X.columns, columns=['importance']).sort_values('importance', ascending=False)
    # feature_importances.to_csv('C:/Users/Sameera/OneDrive - York St John University/MYPROJECT/processedData/KNN_feature_importances.csv')

parser = argparse.ArgumentParser()
parser.add_argument("--sample_size", type=int, default=1)
parser.add_argument("--train_window", type=int, default=3)
parser.add_argument("--forecast_horizon", type=int, default=6)
parser.add_argument("--method", type=str, default="fee")

args = parser.parse_args()
runKNN(args)
