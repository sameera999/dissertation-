import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm 
from metrics import ErrorMetrics
import argparse
from fashionDataset import FashionDataSet;
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Seq2SeqLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out


def runLSTM(args): 
    # Parameters
    num_epochs = 100
    hidden_size = 50
    num_layers = 2
    input_size = 1
    seq_length = 1  # sequence length for LSTM   
    output_size = args.forecast_horizon  # match forecast_horizon

    # Prepare data
    fd = FashionDataSet()
    er = ErrorMetrics()
    X, y = fd.frame_series(args.train_window,args.forecast_horizon,args.method, args.sample_size)
    X = X.values
    y = y.values
    pca = PCA(0.95)
    x_pca = pca.fit_transform(X)
    X = x_pca

    # Reshape data and convert to PyTorch tensors
    X = torch.from_numpy(X.reshape(X.shape[0], X.shape[1], 1)).float()
    y = torch.from_numpy(y.reshape(y.shape[0], args.forecast_horizon)).float()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # LSTM model
    model = Seq2SeqLSTM(input_size, hidden_size, num_layers, output_size)

    # Loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in tqdm(range(num_epochs), desc="LSTM Training"):
        model.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    # Test
    y_pred = model(X_test)
    # Convert tensors to numpy arrays for error calculations
    y_test_np = y_test.detach().numpy()
    y_pred_np = y_pred.detach().numpy()
    er.calculate_errors("LSTM",y_test_np,y_pred_np)


parser = argparse.ArgumentParser()
parser.add_argument("--sample_size", type=int, default=1)
parser.add_argument("--train_window", type=int, default=3)
parser.add_argument("--forecast_horizon", type=int, default=6)
parser.add_argument("--method", type=str, default="fee")

args = parser.parse_args()
runLSTM(args)