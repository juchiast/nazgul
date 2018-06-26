from __future__ import print_function
import argparse
import pandas as pd
import seaborn as sn
sn.set()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
# from conv_simple import Net # change conv_moreconv_relu to something else
import linear_models
import conv_models

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

accuracy = []
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
            print('Train Epoch: {} [{:6d}/{:6d} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), end='\r')

def test(args, model, device, test_loader, accuracies, failed=None):
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
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
#     for x in confusion:
#         print(*x, sep = '\t')
#     cm = pd.DataFrame(confusion, index = list(range(10)), columns = list(range(10)))
#     f, ax = plt.subplots(figsize = (10, 7))
#     sn.heatmap(cm, annot = True, ax = ax)
#     plt.show()
    accuracies.append(correct / len(test_loader.dataset))


class Args:
    def __init__(self):
        self.no_cuda = False
        self.seed = 1
        self.batch_size = 64
        self.test_batch_size = 64
        self.epochs = 20
        self.lr = 0.01
        self.log_interval = 10
        self.momentum = 0.5
        self.save_path = './saved/'
        self.data_path = './data_mnist'

accuracieses = []
def foo(Model, Name):
    print(Name)
    args = Args()

    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}

    transformer = transforms.Compose([
        transforms.Lambda(to_tensor),
    ])

    model = Model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    test_loader = torch.utils.data.DataLoader(
        datasets.EMNIST(args.data_path, split='mnist', train=False, transform=transformer, download = True),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        model.load_state_dict(torch.load(args.save_path + Name, map_location=lambda storage, loc: storage))
        failed = {}
        test(args, model, device, test_loader, failed)
        for k, v in failed.items():
            im = Image.new(mode='L', size=(len(v) * 28, 28))
            for i, f in enumerate(map(lambda h: hashmap.get(h, None), v)):
                if f != None:
                    im.paste(f, (i * 28, 0))
            im.save('./result/' + k + '.png')
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.EMNIST(args.data_path, split='mnist', train=True, transform=transformer, download=True),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        accuracies = []
        for epoch in range(args.epochs):
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, test_loader, accuracies)

        torch.save(model.state_dict(), args.save_path + Name)
        accuracieses.append((accuracies, Name))
        plt.plot(accuracies)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
#         plt.show()
        plt.title(Name)
        plt.savefig(Name)
        plt.clf()


if __name__ == '__main__':
    for Model, Name in linear_models.linear_models:
        foo(Model, Name)
    accuracieseses = [x[0] for x in accuracieses]
    names = [x[1] for x in accuracieses]
    for x in accuracieseses:
        plt.plot(x)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(names)
    plt.savefig('summary linear')
