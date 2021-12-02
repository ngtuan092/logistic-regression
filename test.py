from model import Model2
import torch.nn.functional as F
from dataLoader import test_set
import torch
from torch.utils.data import DataLoader
def test(model):
    print(len(test_set))
    test_loader = DataLoader(test_set)
    result = [model.batch_eval(batch) for batch in test_loader]
    result = model.eval(result)
    print(result)

if __name__ == "__main__":
    try:
        model = Model2(784, 1000, 10, F.relu)
        model.load_state_dict(torch.load('model2'))
        test(model)
    except:
        print("Train the model first")

        
    
