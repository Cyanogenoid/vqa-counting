import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

import counting


class Net(nn.Module):
    def __init__(self, cf):
        super(Net, self).__init__()
        self.cf = cf
        self.counter = counting.Counter(cf, already_sigmoided=True)
        self.classifier = nn.Linear(cf + 1, cf + 1)
        init.eye(self.classifier.weight)

    def forward(self, a, b):
        x = self.counter(b, a)
        return self.classifier(x)


class Baseline(nn.Module):
    def __init__(self, cf):
        super(Baseline, self).__init__()
        self.cf = cf
        self.classifier = nn.Linear(cf + 1, cf + 1)
        self.dummy = counting.Counter(cf, already_sigmoided=True)
        init.eye(self.classifier.weight)

    def forward(self, a, b):
        x = a.sum(dim=1, keepdim=True)
        x = self.dummy.to_one_hot(x)
        return self.classifier(x)
