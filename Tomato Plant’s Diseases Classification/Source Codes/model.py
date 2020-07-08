import torch.nn as nn
import torch.nn.functional as F
import torch


# model MLP with 4 layers (3 Hidden Layer), parameters can be changed to intended values  (DEFAULT MODEL)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features = 519, out_features = 1000)
        self.fc2 = nn.Linear(in_features = 1000, out_features = 512)
        self.fc3 = nn.Linear(in_features = 512, out_features = 200)
        self.fc4 = nn.Linear(in_features = 200, out_features = 10)
        

    def forward(self, x):
        #print(np.max(x))
        x = F.relu(self.fc1(x))  # use activation function to add non-linearity
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

"""
# SIGMOID
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features = 519, out_features = 1000)
        self.fc2 = nn.Linear(in_features = 1000, out_features = 512)
        self.fc3 = nn.Linear(in_features = 512, out_features = 200)
        self.fc4 = nn.Linear(in_features = 200, out_features = 10)
        

    def forward(self, x):
        #print(np.max(x))
        x = torch.sigmoid(self.fc1(x))  # use activation function to add non-linearity
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x

"""
"""
# LESS NEURONS!
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features = 519, out_features = 700)
        self.fc2 = nn.Linear(in_features = 700, out_features = 350)
        self.fc3 = nn.Linear(in_features = 350, out_features = 100)
        self.fc4 = nn.Linear(in_features = 100, out_features = 10)
        

    def forward(self, x):
        #print(np.max(x))
        x = F.relu(self.fc1(x))  # use activation function to add non-linearity
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
"""
"""
# 3 Layers Model with 2 Hidden layers
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features = 519, out_features = 1000)
        self.fc2 = nn.Linear(in_features = 1000, out_features = 512)
        self.fc3 = nn.Linear(in_features = 512, out_features = 10)
        

    def forward(self, x):
        #print(np.max(x))
        x = F.relu(self.fc1(x))  # use activation function to add non-linearity
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
"""