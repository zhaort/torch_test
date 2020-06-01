import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as Data

EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.001
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(root='./mnist', train=True,
                                        transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist', train=False, transform=torchvision.transforms.ToTensor())
text_x = test_data.data.type(torch.FloatTensor)[:2000]/255.
test_y = test_data.targets.numpy().squeeze()[:2000]