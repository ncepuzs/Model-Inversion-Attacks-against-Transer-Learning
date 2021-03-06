from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchvision.datasets as dsets
from torchvision.datasets import ImageFolder
import os
from data import Delete_0_Dataset
from model import Classifier

from sklearn.model_selection import train_test_split

# Training settings
parser = argparse.ArgumentParser(description='Adversarial Model Inversion Demo')
parser.add_argument('--batch-size', type=int, default=128, metavar='')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='')
parser.add_argument('--epochs', type=int, default=100, metavar='')
parser.add_argument('--lr', type=float, default=0.01, metavar='')
parser.add_argument('--momentum', type=float, default=0.5, metavar='')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='')
parser.add_argument('--log-interval', type=int, default=10, metavar='')

parser.add_argument('--nz', type=int, default=530)
parser.add_argument('--num_workers', type=int, default=1, metavar='')
parser.add_argument('--path_out', type=str, default='../Results/out/')


def train(classifier, log_interval, device, data_loader, optimizer, epoch):
    classifier.train()
    try:
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = classifier(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format( epoch, batch_idx * len(data),
                                                                    len(data_loader.dataset), loss.item()))
    except BrokenPipeError:
        pass

def test(classifier, device, data_loader):
    classifier.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = classifier(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest classifier: Average loss: {:.6f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))
    return correct / len(data_loader.dataset)

def main():
    args = parser.parse_args()
    print("================================")
    print(args)
    print("================================")
    os.makedirs('../Results', exist_ok=True)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    torch.manual_seed(args.seed)

    # FaceScrub Dataset
    transform = transforms.Compose([transforms.Resize((32,32)),
                                # transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])
                               ])
    
    train_dataset = dsets.MNIST(root = './data/',
                            train = True, 
                            transform = transform,
                            download = True)
    print("len of train_dataset:", len(train_dataset))

    train_dataset_del0 = Delete_0_Dataset(train_dataset)

    test_dataset = dsets.MNIST(root = './data/',
                           train = False, 
                           transform = transform)
    test_dataset_del0 = Delete_0_Dataset(test_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset_del0, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset_del0, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    classifier = nn.DataParallel(Classifier(nz=args.nz)).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True)

    best_cl_acc = 0
    best_cl_epoch = 0

    # Train classifier
    for epoch in range(1, args.epochs + 1):
        train(classifier, args.log_interval, device, train_loader, optimizer, epoch)
        cl_acc = test(classifier, device, test_loader)

        if cl_acc > best_cl_acc:
            best_cl_acc = cl_acc
            best_cl_epoch = epoch
            state = {
                'epoch': epoch,
                'model': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_cl_acc': best_cl_acc,
            }
            torch.save(state, args.path_out+'classifier.pth')

    print("Best classifier: epoch {}, acc {:.4f}".format(best_cl_epoch, best_cl_acc))

if __name__ == '__main__':
    main()
