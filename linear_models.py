import torch.nn as nn
import torch.nn.functional as F

class linear_simple(nn.Module):
    def __init__(self):
        super(linear_simple, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc(x)
        return F.log_softmax(x, dim = 1)

class linear_sigmoid(nn.Module):
    def __init__(self):
        super(linear_sigmoid, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc(x)
        x = F.sigmoid(x)
        return F.log_softmax(x, dim = 1)

class linear_relu(nn.Module):
    def __init__(self):
        super(linear_relu, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc(x)
        x = F.relu(x)
        return F.log_softmax(x, dim = 1)

class linear_leakyrelu(nn.Module):
    def __init__(self):
        super(linear_leakyrelu, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc(x)
        x = F.leaky_relu(x)
        return F.log_softmax(x, dim = 1)

class linear_elu(nn.Module):
    def __init__(self):
        super(linear_elu, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc(x)
        x = F.elu(x)
        return F.log_softmax(x, dim = 1)

class linear_swish(nn.Module):
    def __init__(self):
        super(linear_swish, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc(x)
        x = x * F.sigmoid(x)
        return F.log_softmax(x, dim = 1)

linear_models = [(linear_simple, 'simple softmax regression'),
                 (linear_sigmoid, 'softmax regression with sigmoid activation'),
                 (linear_relu, 'softmax regression with relu activation'),
                 (linear_leakyrelu, 'softmax regression with leaky relu activation'),
                 (linear_elu, 'softmax regression with elu activation'),
                 (linear_swish, 'softmax regression with swish activation')]
