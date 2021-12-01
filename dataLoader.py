from torch.utils import data
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, dataset, random_split
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

BATCH_SIZE = 128

data_set = MNIST(root='data/', train=True, transform=ToTensor(), download=True)
test_set = MNIST(root='data/', train=False, transform=ToTensor())
if __name__ == "__main__":
    image, labels = data_set[0]
    image = image.reshape(28, 28)
    image2, a = data_set[1]
    image2 = image2.reshape(28, 28)
    plt.figure(1)
    plt.imshow(image, cmap='gray')
    
    plt.figure(2)
    plt.imshow(image2, cmap='gray')
    plt.show() # show the image

train_set, valid_set = random_split(data_set, [50000, 10000])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE) 

