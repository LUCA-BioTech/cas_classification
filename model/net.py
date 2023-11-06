from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
    correct = 0
    _, predicted = torch.max(outputs, 1)
    correct += (predicted == labels).sum().item()
    return correct / len(labels)

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}