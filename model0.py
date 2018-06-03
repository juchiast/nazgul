from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import sys
import hashlib
from PIL import Image

hashmap = {}

def to_tensor(image):
    x = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])(image)
    hashmap[hashlib.md5(x.numpy()).hexdigest()] = image
    return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), end='\r')

def test(args, model, device, test_loader, failed=None):
    model.eval()
    test_loss = 0
    correct = 0
    confusion = [[0] * 10 for i in range(10)]
    with torch.no_grad():
        for _data, target in test_loader:
            data, target = _data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            for i, p in enumerate(pred.eq(target.view_as(pred))):
                confusion[pred[i].item()][target[i].item()] += 1
                if p.item() == 0:
                    if failed is not None:
                        name = str(target[i].item()) + '_' + str(pred[i].item())
                        h = hashlib.md5(_data[i].numpy()).hexdigest()
                        if name in failed:
                            failed[name].append(h)
                        else:
                            failed[name] = [h]
                else:
                    correct += 1

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('Confusion matrix:')
    for x in confusion:
        print(*x, sep = '\t')


class Args:
    def __init__(self):
        self.no_cuda = False
        self.seed = 1
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 20
        self.lr = 0.01
        self.log_interval = 10
        self.momentum = 0.5
        self.save_path = './saved/model0'
        self.data_path = './data_mnist'

def main():
    args = Args()

    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 32, 'pin_memory': True} if use_cuda else {}

    transformer = transforms.Compose([
        transforms.Lambda(to_tensor),
    ])

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_path, train=False, transform=transformer),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    print(sys.argv)
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        model.load_state_dict(torch.load(args.save_path))
        failed = {}
        test(args, model, device, test_loader, failed)
        for k, v in failed.items():
            im = Image.new(mode='L', size=(len(v) * 28, 28))
            for i, f in enumerate(map(lambda h: hashmap[h], v)):
                im.paste(f, (i * 28, 0))
            im.save('./result/' + k + '.png')
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_path, train=True, transform=transformer),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        for epoch in range(args.epochs):
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, test_loader)

        torch.save(model.state_dict(), args.save_path)


if __name__ == '__main__':
    main()
