import torch.nn as nn
import torch.nn.functional as F


# model FCN with 3 layers, parameters can be changed to intended values
class FCN_3L(nn.Module):
    def __init__(self):
        super(FCN_3L, self).__init__()
        self.fc1 = nn.Linear(in_features = 150*150, out_features = 300)
        self.fc2 = nn.Linear(in_features = 300, out_features = 120)
        self.fc3 = nn.Linear(in_features = 120, out_features = 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # use activation function to add non-linearity
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# FRUIT DATASETI İÇİN
class FCN_fruit(nn.Module):
    def __init__(self):
        super(FCN_fruit, self).__init__()
        self.fc1 = nn.Linear(in_features = 50*50, out_features = 500)
        self.fc2 = nn.Linear(in_features = 500, out_features = 250)
        self.fc3 = nn.Linear(in_features = 250, out_features = 120)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))  # use activation function to add non-linearity
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x