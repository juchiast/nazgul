import torch.nn as nn
import torch.nn.functional as F

class LRNet(nn.Module):
    def __init__(self):
        super(LRNet, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc(x)
        x = x * F.sigmoid(x)
        return F.log_softmax(x, dim = 1)
