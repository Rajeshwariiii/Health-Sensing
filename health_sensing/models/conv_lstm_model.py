import torch
import torch.nn as nn

class ConvLSTMModel(nn.Module):
    def __init__(self, input_channels=3, hidden_size=64, num_layers=1):
        super(ConvLSTMModel, self).__init__()
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.2)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, 
                           num_layers=num_layers, batch_first=True, dropout=0.2)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        # Convolutional layers
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Reshape for LSTM (batch_size, seq_len, features)
        x = x.transpose(1, 2)  # (batch, time, channels)
        
        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the last output
        out = lstm_out[:, -1, :]
        
        # Fully connected layer
        out = self.fc(out)
        
        return out