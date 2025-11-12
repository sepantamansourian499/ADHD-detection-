import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGSmallCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2, stride=1)  # keep C,T same

        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, padding=3, stride=(1, 2))  # T/2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=(1, 2))  # T/2
        self.conv4 = nn.Conv2d(32, 64, kernel_size=11, padding=5, stride=(1, 2))  # T/2

        # self.conv5 = nn.Conv2d(192, 128, kernel_size=3)  # T/4 overall

        self.fc3 = nn.Linear(192, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))       # [B,32, C, T]

        b2 = F.relu(self.conv2(x1))      # [B,64, C, T/2]
        b3 = F.relu(self.conv3(x1))      # [B,64, C, T/2]
        b4 = F.relu(self.conv4(x1))      # [B,64, C, T/2]

        x_cat = torch.cat([b2, b3, b4], dim=1)  # [B,192, C, T/2]

        x5 = F.adaptive_avg_pool2d(x_cat, (1, 1)).squeeze(-1).squeeze(-1)  # [B,128]
        x5 = F.relu(self.fc3(x5))   # [B,128, C, T/4]
        x5 = F.relu(self.fc1(x5))        # [B,64]
        return self.fc2(x5)
