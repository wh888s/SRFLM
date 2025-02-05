#!/usr/bin/env python
import os
import numpy as np
from torchvision import datasets, transforms
import argparse
import torch
from torch import nn, autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
from sklearn import metrics
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--r_epochs', type=int, default=50, help="recurrent rounds of training")
    parser.add_argument('--s_epochs', type=int, default=2, help="recurrent rounds of training")
    parser.add_argument('--num_users', type=int, default=4, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_false', help='aggregation over all clients')
    args,_ = parser.parse_known_args()
    return args

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    

class ServerUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class CNNMnist(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64))
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(3136, 256)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 3136)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]   
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


def noniid(dataset, num_users):
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(60000)
    labels = dataset.train_labels.numpy()
    # sort labels
    idxs_labels = np.vstack((idxs, labels))                                                                   
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()] 
    idxs = idxs_labels[0,:]
    for i in range(num_users):
        a=idxs_labels[:,idxs_labels[1]==i]
        print(a)
        dict_users[i]=a[0]
    print(dict_users)
    return dict_users

def noniid_0689(dataset, num_users):
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(60000)
    labels = dataset.train_labels.numpy()
    # sort labels
    idxs_labels = np.vstack((idxs, labels))

    for digit in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        digit_idxs = idxs_labels[:, idxs_labels[1, :] == digit]
        digit_idxs = digit_idxs[0]

        np.random.shuffle(digit_idxs)

        digit_idxs = digit_idxs[:int(0.8 * len(digit_idxs))]

        for i in range(num_users):
            user_idxs = digit_idxs[i::num_users]
            dict_users[i] = np.concatenate([dict_users[i], user_idxs])

    dict_users[0] = np.concatenate([dict_users[0], idxs_labels[0, idxs_labels[1, :] == 0]])

    dict_users[1] = np.concatenate([dict_users[1], idxs_labels[0, idxs_labels[1, :] == 6]])

    dict_users[2] = np.concatenate([dict_users[2], idxs_labels[0, idxs_labels[1, :] == 8]])

    dict_users[3] = np.concatenate([dict_users[3], idxs_labels[0, idxs_labels[1, :] == 9]])

    print(dict_users)
    return dict_users

########################################################### RFL ######################################################################
args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
dict_users = noniid_0689(dataset_train, args.num_users)
img_size = dataset_train[0][0].shape

args.model == 'cnn'
net_glob = CNNMnist(args=args).to(args.device)
print(net_glob)
net_glob.train()

# copy weights
w_glob = net_glob.state_dict()
        
# training
loss_train = []
loss_train_global = []
cv_loss, cv_acc = [], []
val_loss_pre, counter = 0, 0
net_best = None
best_loss = None
val_acc_list, net_list = [], []
w_locals = [w_glob for i in range(args.num_users)]

data0=torch.utils.data.Subset(dataset_train,dict_users[0])
data1=torch.utils.data.Subset(dataset_train,dict_users[1])
data2=torch.utils.data.Subset(dataset_train,dict_users[2])
data3=torch.utils.data.Subset(dataset_train,dict_users[3])

epoch=0
idxs_users = np.random.choice(range(args.num_users), args.num_users, replace=False)
for n in range(int(args.r_epochs/(args.s_epochs*args.num_users))):       
    for server_users in idxs_users:
        for i in range(args.s_epochs):
            epoch+=1
            loss_locals = []
            for idx in idxs_users:
                if idx != server_users:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                    w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    if args.all_clients:
                        w_locals[idx] = copy.deepcopy(w)
                    else:
                        w_locals.append(copy.deepcopy(w))
                    loss_locals.append(copy.deepcopy(loss))

            w_glob = FedAvg(w_locals)
            net_glob.load_state_dict(w_glob)
            server = ServerUpdate(args=args, dataset=dataset_train, idxs=dict_users[server_users])
            w_server, loss_server = server.train(net=copy.deepcopy(net_glob).to(args.device))
            net_glob.load_state_dict(w_server)

            loss_avg = sum(loss_locals) / len(loss_locals)
            loss_train.append(loss_avg)
            loss_train_global.append(loss_server)
            print('Round {:3d}, Average loss {:.5f}, Global loss: {:.5f}'.format(epoch, loss_avg, loss_server))
            
plt.figure()
plt.plot(range(len(loss_train)), loss_train)
plt.ylabel('local train_loss')
save_path = "./save/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
plt.savefig(os.path.join(save_path, 'local_{}_{}_{}_{}_non-iid.png'.format(args.dataset, args.model, args.r_epochs, args.num_users)))
#
plt.figure()
plt.plot(range(len(loss_train_global)), loss_train_global)
plt.ylabel('global train_loss')
plt.savefig(os.path.join(save_path, 'global_{}_{}_{}_{}_non-iid.png'.format(args.dataset, args.model, args.r_epochs, args.num_users)))

net_glob.eval()
acc_test0, loss_test0 = test_img(net_glob, data0, args)
acc_test1, loss_test1 = test_img(net_glob, data1, args)
acc_test2, loss_test2 = test_img(net_glob, data2, args)
acc_test3, loss_test3 = test_img(net_glob, data3, args)
print("Testing accuracy: client1:{:.2f}, client2:{:.2f}, client3:{:.2f}, client4:{:.2f}".format(acc_test0,acc_test1,acc_test2,acc_test3))


