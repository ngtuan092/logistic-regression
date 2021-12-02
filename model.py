import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.eval import Accuracy


class Model(nn.Module):
    def __init__(self, num_input, num_output):
        super().__init__()
        self.num_input = num_input
        self.num_output = num_output
        self.linear = nn.Linear(num_input, num_output)

    def forward(self, input):
        # one-layer network
        input = input.reshape(-1, self.num_input)
        out = self.linear(input)
        return out

    def loss_calculate(self, batch):
        xb, yb = batch
        out = self(xb)
        loss = F.cross_entropy(out, yb)
        return loss

    def batch_eval(self, batch):
        xb, yb = batch
        out = self(xb)
        loss = F.cross_entropy(out, yb)
        acc = Accuracy(out, yb)
        return {'val_loss': loss, 'val_acc': acc}

    def eval(self, batch_results):
        batch_losses = [x['val_loss'] for x in batch_results]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in batch_results]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}


class Model2(nn.Module):
    """
    More complex model try to increase accuracy
    """

    def __init__(self, num_input, num_hidden, num_output, activation_fn):
        super().__init__()
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.activation_fn = activation_fn
        self.linear1 = nn.Linear(num_input, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_output)

    def forward(self, input):
        # two-layer network
        input = input.reshape(-1, self.num_input)
        hidden_out = self.activation_fn(self.linear1(input))
        out = self.activation_fn(self.linear2(hidden_out))
        return out

    def loss_calculate(self, batch):
        xb, yb = batch
        out = self(xb)
        loss = F.cross_entropy(out, yb)
        return loss

    def batch_eval(self, batch):
        xb, yb = batch
        out = self(xb)
        loss = F.cross_entropy(out, yb)
        acc = Accuracy(out, yb)
        return {'val_loss': loss, 'val_acc': acc}

    def eval(self, batch_results):
        batch_losses = [x['val_loss'] for x in batch_results]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in batch_results]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
