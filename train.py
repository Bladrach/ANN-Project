import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import model

from custom_dataset_loader import MyCustomDataset, MyCustomFruitDataset
from torch.utils.data import Dataset, DataLoader


batch_size = 16
max_epoch = 100
#data_path = 'C:\\Users\\Mehmet\\Desktop\\ANN\\img_dataset\\edged_intel'
data_path = 'C:\\Users\\Mehmet\\Desktop\\ANN\\img_dataset\\fruits\\fruits-360'

# Output files
netname = "net"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

"""
# normalization for grayscale
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])
"""
# Initialize the dataset and dataloader
traindataset = MyCustomFruitDataset(data_path = data_path, train = True)
trainloader = DataLoader(traindataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 0)

testdataset = MyCustomFruitDataset(data_path = data_path, train = False)
testloader = DataLoader(testdataset, batch_size = 1, shuffle = True, pin_memory = True, num_workers = 0)

# Define Net
net = model.FCN_fruit()
net = net.to(device)

########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9)

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.
if __name__ == "__main__":
    
    print("Training Started")
    for epoch in range(max_epoch):
        batchiter = 0
        for batch in trainloader:
            
            batchiter += 1
            inputs = batch[0]

            #inputs = inputs.view(-1, 150*150)   INTEL İÇİN
            inputs = inputs.view(-1, 50*50)
            inputs = inputs.to(device)

            label = batch[1].to(device)
        
            outputs = net(inputs.float())   
            optimizer.zero_grad()    
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
                
            print("TRAIN","Epoch:",epoch+1, "Data-Num:",batchiter, "Loss:",loss.item(), " label: ", label.tolist())

        if epoch % 5 == 4:
            torch.save(net.state_dict(), "./saved_models/" + netname + "_epoch_%d"%(epoch) + ".pth")
    print('Finished Training')
