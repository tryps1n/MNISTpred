import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt

class CTDataset(Dataset):
    def __init__(self, data):
        self.x = data.data.float() / 255.0 # change all values from 0 to 1
        self.y = F.one_hot(data.targets, num_classes=10).float() # form classes for each outcome
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]
    
class MNISTneuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28**2, 100)
        self.layer2 = nn.Linear(100, 50)
        self.layer3 = nn.Linear(50, 10)
        self.R = nn.ReLU()
    def forward(self, x):
        x = x.view(-1, 28**2)
        x = self.R(self.layer1(x))
        x = self.R(self.layer2(x))
        x = self.layer3(x)
        return x.squeeze() 

def train_model(dl, neunet, n_epochs=20):
    opt = SGD(neunet.parameters(), lr=0.01)
    L = nn.CrossEntropyLoss()

    losses = []
    epochs = []

    for epoch in range(n_epochs):
        print(f'epoch number {epoch}')
        N = len(dl)
        for i, (x, y) in enumerate(dl):
            opt.zero_grad()
            loss_value = L(neunet(x), y)
            loss_value.backward()
            opt.step()

            epochs.append(epoch+i/N)
            losses.append(loss_value.item())
    
    return np.array(epochs), np.array(losses)
