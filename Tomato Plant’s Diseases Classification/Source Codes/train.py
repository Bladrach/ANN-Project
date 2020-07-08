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
from custom_dataset import CustomDataset
from torch.utils.data import Dataset, DataLoader
import time
import math


batch_size = 32
max_epoch = 11
data_path = "C:\\Users\\Mehmet\\Desktop\\yeniANN"

# Output files
netname = "net"

# For using GPU if it exists
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = 'cuda'

# Initialize the dataset and dataloader
traindataset = CustomDataset(data_path = data_path, train = True, val = False)
trainloader = DataLoader(traindataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 0)

valdataset = CustomDataset(data_path = data_path, train = False, val = True)
valloader = DataLoader(traindataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 0)

testdataset = CustomDataset(data_path = data_path, train = False, val = False)
testloader = DataLoader(testdataset, batch_size = 1, shuffle = True, pin_memory = True, num_workers = 0)

# Define Net
net = model.MLP()
net = net.to(device)

# 3. Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(params = net.parameters(), lr = 1e-3, momentum = 0.9)
optimizer = optim.Adam(params = net.parameters(), lr = 1e-3)

train_loss_list = []
val_loss_list = []
epoch_list = []

# 4. Train the network
if __name__ == "__main__":
    
    print("Training Started")

    train_loss = 0
    val_loss = 0
    since = time.time()
    for epoch in range(max_epoch):
        batchiter = 0

        for batch in trainloader:
            net.train()  # starting with train mode / also, to return train mode from evaluation mode in validation part
            batchiter += 1
            inputs = batch[0]
            inputs = inputs.view(-1, 519)
            inputs = inputs.to(device)
            label = batch[1].to(device)
            outputs = net(inputs)   
            optimizer.zero_grad()    
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            if (batchiter % ((len(traindataset) // batch_size) + 1) == 0):  # calculate train loss after an epoch ends
                train_loss = train_loss / batchiter
                train_loss_list.append(train_loss)

            print("TRAIN", "Epoch:", epoch, "Data-Num:", batchiter, "Loss:", loss.item(), " label: ", label.tolist())

            if (batchiter % ((len(traindataset) // batch_size) + 1) == 0):
                net.eval()  # changing train mode to evaluation mode for validation
                print("Processing validation... Please wait...")
                with torch.no_grad():
                    for data in valloader:
                        inputs = data[0].view(-1, 519)
                        #print(inputs.shape)
                        inputs = inputs.to(device)
                        label = data[1].to(device)
                        outputs = net(inputs)
                        loss = criterion(outputs, label)
                        val_loss += loss.item()
                val_loss = val_loss / ((len(valdataset) // batch_size) + 1)  # calculate val loss after an epoch ends
                val_loss_list.append(val_loss)
                print("Epoch:", epoch, "VAL Loss:", val_loss, "Train Loss:", train_loss)
   
        epoch_list.append(epoch)
        
        if epoch % 1 == 0:
            torch.save(net.state_dict(), "./saved_models_pv/" + netname + "_epoch_%d"%(epoch) + ".pth")
    time_elapsed = time.time() - since
    print('Finished Training')

    elapsedTimeFile = open('elapsed_time.txt', 'w')  # open a text file in write mode
    elapsedTimeFile.write('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  # elapsed time is written on a text file
    elapsedTimeFile.close()  # close the text file
    
    fig = plt.figure()
    plt.plot(epoch_list, train_loss_list, 'b', label = 'train')
    plt.plot(epoch_list, val_loss_list, 'r', label = 'val')
    plt.title("Loss Graph, Elapsed Time ({:.0f}m {:.0f}s)".format(time_elapsed // 60, time_elapsed % 60))
    plt.grid()
    plt.xlabel("Epoch")
    plt.xticks(range(0, 11, 1))
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("TrainHistory.jpg")
    plt.show()
    