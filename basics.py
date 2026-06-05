import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from lib import CTDataset, MNISTneuralNetwork, train_model

transform = torchvision.transforms.ToTensor()

train_ds = CTDataset(torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform))
test_ds = CTDataset(torchvision.datasets.MNIST(root='data', train=False, download=False, transform=transform))

train_dl = DataLoader(train_ds, batch_size=5)
test_dl = DataLoader(test_ds, batch_size=5)

model = MNISTneuralNetwork()

epoch_data, loss_data = train_model(train_dl, model)

plt.plot(epoch_data, loss_data)

torch.save(model.state_dict(), 'data/mnist_model.pth')

print("finished training, model saved.")

# epoch_data_avg = epoch_data.reshape(20, -1).mean(axis=1)
# loss_data_avg = loss_data.reshape(20, -1).mean(axis=1)

# xs, ys = test_ds[0:2000]
# yhats = model(xs).argmax(axis=1)

# fig, ax = plt.subplots(10, 4, figsize=(10, 15))
# for i in range(40):
#     plt.subplot(10, 4, i+1)
#     plt.imshow(xs[i])
#     plt.title(f'Predicted Digit {yhats[i]}')
# fig.tight_layout()
# plt.show()