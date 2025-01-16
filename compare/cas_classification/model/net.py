
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.metrics import precision_score, recall_score, f1_score

class Net(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.layer1 = nn.Linear(params.embedding_dim, params.hidden_dim)
        self.layer2 = nn.Linear(params.hidden_dim, params.number_of_class)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        return self.layer2(x)


def loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs,labels)

def accuracy(outputs, labels):
    _, predicted = torch.max(torch.tensor(outputs), 1)
    return (predicted == torch.tensor(labels)).sum().item() / len(labels)

def precision(outputs, labels):
    _, predicted = torch.max(torch.tensor(outputs), 1)
    return precision_score(labels, predicted, average=None)  # per-class precision

def recall(outputs, labels):
    _, predicted = torch.max(torch.tensor(outputs), 1)
    return recall_score(labels, predicted, average=None)  # per-class recall

def f1(outputs, labels):
    _, predicted = torch.max(torch.tensor(outputs), 1)
    return f1_score(labels, predicted, average=None)  # per-class F1

metrics = {
    'accuracy': accuracy,
    #'precision': precision,
    #'recall': recall,
    #'f1': f1
}
