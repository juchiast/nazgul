import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=5)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(480, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x) # 10x26x26
        x = F.sigmoid(F.max_pool2d(self.conv2(x), 2)) # 10x24x24 -> 20x12x12
        x = F.sigmoid(F.max_pool2d(self.conv_drop(self.conv3(x)), 2)) # 20x8x8 -> 30x4x4
        x = x.view(-1, 480)
        x = F.sigmoid(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.sigmoid(self.fc2(x))
        return F.log_softmax(x, dim=1)

