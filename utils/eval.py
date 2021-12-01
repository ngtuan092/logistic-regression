import torch

def Accuracy(out, true_labels):
    pred_labels = torch.argmax(out, dim=1)
    return torch.tensor(torch.sum(pred_labels == true_labels).item() / len(pred_labels))

