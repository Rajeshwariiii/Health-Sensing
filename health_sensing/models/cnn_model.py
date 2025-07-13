import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, input_channels=3):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 480, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)