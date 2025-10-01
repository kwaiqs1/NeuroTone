# audio/learn/models.py
import torch, torch.nn as nn, torch.nn.functional as F

class TinyCNN(nn.Module):
    def __init__(self, n_mels: int, n_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d((2,2))
        self.head  = nn.Linear(64, n_classes)

    def forward(self, x):  # x: [B,1,M,T]
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.mean(dim=[2,3])         # GAP
        return self.head(x)
