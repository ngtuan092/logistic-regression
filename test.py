from model import Model
import torch.nn.functional as F
from dataLoader import test_set
import torch
from torch.utils.data import DataLoader
def main(model):
    print(len(test_set))
    test_loader = DataLoader(test_set)
    result = [model.batch_eval(batch) for batch in test_loader]
    result = model.eval(result)
    print(result)

if __name__ == "__main__":
    try:
        model = Model(784, 10)
        model.load_state_dict(torch.load('mnist_model'))
        main(model)
    except:
        print("Train the model first")

        
    
