import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm 
from metrics import ErrorMetrics
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

class FashionDataSet:    
    # Function to frame the time series data
    def frame_series(self, train_window=3, forecast_horizon=1, method='nor', sample_size=1):
        df = pd.read_csv("C:/Users/Sameera/OneDrive - York St John University/MYPROJECT/processedData/processedSales.csv")
        df = df.sort_values(by="release_date")
        n_samples = int(len(df) * sample_size)        
        df = df[:n_samples]
        
        X, y = [], []
        sales_columns = [f'w{i}_sales' for i in range(1, 13)]
            
        for (_, row) in tqdm(df.iterrows(), total=len(df)):
            sales = row[sales_columns].values.astype(float)  # Convert sales to float
            
            for j in range(len(sales) - train_window - forecast_horizon + 1):
                features = list(sales[j : j + train_window])  
                target = sales[j + train_window : j + train_window + forecast_horizon]
                X.append(features)
                y.append(target)
                
        return np.array(X), np.array(y)

# Parameters
num_epochs = 100
hidden_size = 50
num_layers = 2
input_size = 1
seq_length = 1  # sequence length for LSTM
train_window = 2
forecast_horizon = 2
method = "nor"
sample_size = 1
output_size = forecast_horizon  # match forecast_horizon

# Prepare data
fd = FashionDataSet()
er = ErrorMetrics()
X, y = fd.frame_series(train_window, forecast_horizon, method, sample_size)

# Reshape data and convert to PyTorch tensors
X = torch.from_numpy(X.reshape(X.shape[0], X.shape[1], 1)).float()
y = torch.from_numpy(y.reshape(y.shape[0], forecast_horizon)).float()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# LSTM model
model = Seq2SeqLSTM(input_size, hidden_size, num_layers, output_size)

# Loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(num_epochs):
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