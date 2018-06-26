import torch.nn as nn
import torch.nn.functional as F

class conv_simple(nn.Module):
    def __init__(self):
        super(conv_simple, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class conv_moreconv(nn.Module):
    def __init__(self):
        super(conv_moreconv, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=5)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(480, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x) # 10x26x26
        x = F.max_pool2d(self.conv2(x), 2) # 10x24x24 -> 20x12x12
        x = F.max_pool2d(self.conv_drop(self.conv3(x)), 2) # 20x8x8 -> 30x4x4
        x = x.view(-1, 480)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class conv_moreconv_sigmoid(nn.Module):
    def __init__(self):
        super(conv_moreconv_sigmoid, self).__init__()
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

class conv_moreconv_relu(nn.Module):
    def __init__(self):
        super(conv_moreconv_relu, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=5)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(480, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x) # 10x26x26
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) # 10x24x24 -> 20x12x12
        x = F.relu(F.max_pool2d(self.conv_drop(self.conv3(x)), 2)) # 20x8x8 -> 30x4x4
        x = x.view(-1, 480)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

class conv_moreconv_leakyrelu(nn.Module):
    def __init__(self):
        super(conv_moreconv_leakyrelu, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=5)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(480, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x) # 10x26x26
        x = F.leaky_relu(F.max_pool2d(self.conv2(x), 2)) # 10x24x24 -> 20x12x12
        x = F.leaky_relu(F.max_pool2d(self.conv_drop(self.conv3(x)), 2)) # 20x8x8 -> 30x4x4
        x = x.view(-1, 480)
        x = F.leaky_relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.leaky_relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

class conv_moreconv_elu(nn.Module):
    def __init__(self):
        super(conv_moreconv_elu, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=5)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(480, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x) # 10x26x26
        x = F.elu(F.max_pool2d(self.conv2(x), 2)) # 10x24x24 -> 20x12x12
        x = F.elu(F.max_pool2d(self.conv_drop(self.conv3(x)), 2)) # 20x8x8 -> 30x4x4
        x = x.view(-1, 480)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.elu(self.fc2(x))
        return F.log_softmax(x, dim=1)

def swish(x):
    return x * F.sigmoid(x)
class conv_moreconv_swish(nn.Module):
    def __init__(self):
        super(conv_moreconv_swish, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=5)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(480, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x) # 10x26x26
        x = swish(F.max_pool2d(self.conv2(x), 2)) # 10x24x24 -> 20x12x12
        x = swish(F.max_pool2d(self.conv_drop(self.conv3(x)), 2)) # 20x8x8 -> 30x4x4
        x = x.view(-1, 480)
        x = swish(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = swish(self.fc2(x))
        return F.log_softmax(x, dim=1)
                
conv_models = [(conv_simple, 'simple convolutional neural network'),
               (conv_moreconv_sigmoid, 'convolutional neural network with sigmoid activation'),
               (conv_moreconv, 'cnn with more convolutional layer'),
               (conv_moreconv_relu, 'convolutional neural network with relu activation'),
               (conv_moreconv_leakyrelu, 'convolutional neural network with leaky relu activation'),
               (conv_moreconv_elu, 'convolutional neural network with elu activation'),
               (conv_moreconv_swish, 'convolutional neural network with swish activation')]
