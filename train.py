from torch.serialization import save
from dataLoader import train_loader, valid_loader
from model import Model2
import matplotlib.pyplot as plt
import torch
NUM_INPUT = 784
NUM_OUTPUT = 10
import torch.nn.functional as F
# init model
model = Model2(NUM_INPUT, 1000,  NUM_OUTPUT, F.relu)
batch_results = [model.batch_eval(batch) for batch in valid_loader]
result0 = model.eval(batch_results)

# train model
def fit(epochs, lr, model, train_loader, valid_loader, opt=torch.optim.SGD):
    optimizer = opt(model.parameters(), lr)
    history = []
    for epoch in range(epochs):
        for batch in train_loader:
            loss = model.loss_calculate(batch)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        batch_results = [model.batch_eval(batch) for batch in valid_loader]
        result = model.eval(batch_results)
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
        history.append(result)
    return history

def save_model(model, model_name):
    torch.save(model.state_dict(), model_name)

if __name__ == '__main__':
    history = fit(20, 0.03, model, train_loader, valid_loader)
    save_model(model, 'model2')
    history = [result0] + history
    accuracies = [result['val_acc'] for result in history]
    plt.figure("Fig.1")
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    loss = [result['val_loss'] for result in history]
    plt.figure("Fig.2")
    plt.plot(loss, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs')
    plt.show()