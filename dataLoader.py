import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, dataset, random_split
from torchvision.transforms import ToTensor
BATCH_SIZE = 128

data_set = MNIST(root='data/', train=True, transform=ToTensor(), download=True)
test_set = MNIST(root='data/', train=False, transform=ToTensor())

train_set, valid_set = random_split(data_set, [50000, 10000])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE) 