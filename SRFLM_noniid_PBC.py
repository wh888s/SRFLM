#!/usr/bin/env python
import os
import numpy as np
from torchvision import transforms
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
import scipy.io as sio
from ResNet_class import resnet18

def args_parser():
    parser = argparse.ArgumentParser()
    # Federated Learning Parameters
    parser.add_argument('--r_epochs', type=int, default=20, help="Total number of training rounds")
    parser.add_argument('--s_epochs', type=int, default=1, help="Number of server rounds")
    parser.add_argument('--num_users', type=int, default=4, help="Number of clients")
    parser.add_argument('--frac', type=float, default=0.1, help="Client sampling fraction")
    parser.add_argument('--local_ep', type=int, default=10, help="Number of local training epochs")
    parser.add_argument('--server_ep', type=int, default=20, help="Number of server training epochs")
    parser.add_argument('--local_bs', type=int, default=16, help="Local batch size")
    parser.add_argument('--server_bs', type=int, default=16, help="Server batch size")
    parser.add_argument('--bs', type=int, default=32, help="Test batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum")
    # Model Parameters
    parser.add_argument('--model', type=str, default='ResNet18', help='Model name')
    parser.add_argument('--norm', type=str, default='batch_norm', help="Normalization method")
    # Other Parameters
    parser.add_argument('--dataset', type=str, default='PBC', help="Dataset name")
    parser.add_argument('--num_classes', type=int, default=8, help="Number of classes")
    parser.add_argument('--data_distribution', type=int, default=1, help="REGULAR=1, CLASS=2, SZIE=3")
    parser.add_argument('--num_channels', type=int, default=3, help="Number of image channels")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID")
    parser.add_argument('--seed', type=int, default=65, help="Random seed")
    args, _ = parser.parse_known_args()
    return args

class MyDataset(Dataset):
    def __init__(self, images, labels, setname):
        self.setname = setname
        self.images = images
        self.labels = labels
        self.transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        image = image.permute(2,1,0)
        img = self.transform(image)
        return img, self.labels[idx]

def load_dataset(data_root):
    data = np.load(data_root)
    train_images = data['train_images']
    train_labels = data['train_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']
    dataset_train = MyDataset(train_images, train_labels, 'train')
    dataset_test = MyDataset(test_images, test_labels, 'test')
    return dataset_train, dataset_test


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        label = int(label[0])
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, client=None):
        self.args = args
        self.client = client
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), 
                                 batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, current_lr):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=current_lr, 
                                  momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # print(labels)
                optimizer.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss)/len(epoch_loss)

class ServerUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, client=None):
        self.args = args
        self.client = client
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs),
                                 batch_size=self.args.server_bs, shuffle=True)

    def train(self, net, current_lr):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=current_lr,
                                  momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.server_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss)/len(epoch_loss)

class CNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64))
        self.pool = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, args.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]   
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def dirichlet_split_noniid(train_labels, alpha, n_clients, state):
    if len(train_labels.shape) > 1:
        train_labels = np.squeeze(train_labels)
    n_classes = train_labels.max() + 1
    np.random.set_state(state)
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    client_idcs = {i: np.array([], dtype='int64') for i in range(n_clients)}
    for c, fracs in zip(class_idcs, label_distribution):
        split_idcs = np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))
        for i, idcs in enumerate(split_idcs):
            client_idcs[i] = np.concatenate([client_idcs[i], idcs])
    return client_idcs

def class_split(dataset_labels, client_classes):
    client_idcs = {client: [] for client in client_classes}
    for cls in np.unique(dataset_labels):
        cls_indices = np.where(dataset_labels == cls)[0]
        np.random.shuffle(cls_indices)
        target_clients = [client for client, classes in client_classes.items() if cls in classes]
        split_indices = np.array_split(cls_indices, len(target_clients))
        for client, indices in zip(target_clients, split_indices):
            client_idcs[client].extend(indices.tolist())
    for client in client_idcs:
        np.random.shuffle(client_idcs[client])
    return client_idcs

def test_img(net_g, dataset, args):
    net_g.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(dataset, batch_size=args.bs)
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = net_g(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    print('Testing | Average loss: {:.4f} | Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


if __name__ == "__main__":
    save_path = "./save-PBC/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    state = np.random.get_state()
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Load dataset
    dataset_train, dataset_test = load_dataset('PBCdataset.npz')
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if args.data_distribution == 1:
        dict_users_train = dirichlet_split_noniid(train_labels=dataset_train.labels, alpha=0.5, n_clients=args.num_users, state=state)
        dict_users_test = dirichlet_split_noniid(train_labels=dataset_test.labels, alpha=0.5, n_clients=args.num_users, state=state)
    elif args.data_distribution == 2:
        client_classes = {0: [0, 1, 2], 1: [3, 4], 2: [5, 6], 3: [7]}
        dict_users_train = class_split(dataset_train.labels.squeeze(), client_classes)
        dict_users_test = class_split(dataset_test.labels.squeeze(), client_classes)
        extract_size = 1600
        for client in dict_users_train:
            client_indices = dict_users_train[client]
            dict_users_train[client] = client_indices[:extract_size]
    elif args.data_distribution == 3:
        dict_users_train = dirichlet_split_noniid(train_labels=dataset_train.labels, alpha=1.0, n_clients=args.num_users, state=state)
        dict_users_test = dirichlet_split_noniid(train_labels=dataset_test.labels, alpha=1.0, n_clients=args.num_users, state=state)
    else:
        raise ValueError("Invalid data distribution option.")

    # idxs_users = np.random.choice(range(args.num_users), args.num_users, replace=False)
    idxs_users = [0, 1, 2, 3]
    print("client1:{:.2f}, client2:{:.2f}, client3:{:.2f}, client4:{:.2f}".format(idxs_users[0], idxs_users[1], idxs_users[2], idxs_users[3]))
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("\n【Class Distribution Report】")
    for client in idxs_users:
        train_indices = dict_users_train[client]
        train_labels = [dataset_train.labels[idx][0] for idx in train_indices]
        unique_train, counts_train = np.unique(train_labels, return_counts=True)

        test_indices = dict_users_test[client]
        test_labels = [dataset_test.labels[idx][0] for idx in test_indices]
        unique_test, counts_test = np.unique(test_labels, return_counts=True)
        
        print(f"\nClient {client}:")
        print("Training Data:")
        print(f"  Classes: {len(unique_train)}, Total samples: {len(train_indices)}")
        print("  Class Details:")
        for cls, count in zip(unique_train, counts_train):
            print(f"    ▸ Class {cls}: {count} samples")
        
        print("Testing Data:")
        print(f"  Classes: {len(unique_test)}, Total samples: {len(test_indices)}")
        print("  Class Details:")
        for cls, count in zip(unique_test, counts_test):
            print(f"    ▸ Class {cls}: {count} samples")

    print("\n【Dataset Summary】")
    print(f"Total Training Samples: {len(dataset_train)}")
    print(f"Total Testing Samples: {len(dataset_test)}")
    print(f"Number of Classes: {args.num_classes}")
    print(f"Sample Shape: {dataset_train[0][0].shape}")
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # net_glob = CNN(args).to(args.device)
    net_glob = resnet18(pretrained=False, num_classes=8-1).to(args.device)
    print(net_glob)
    w_glob = net_glob.state_dict()
    net_best_m = {idx: net_glob.to(args.device) for idx in range(args.num_users)}

    loss_train = []
    loss_train_global = []
    local_weights = [w_glob for _ in range(args.num_users)]

    epoch = 0
    current_lr = args.lr
    initial_rounds = 5
    server_candidates = list(idxs_users)
    best_loss = {idx: 1 for idx in range(args.num_users)}
    for rnd in range(args.r_epochs):
        if rnd >= initial_rounds:
            total_data = sum([len(dict_users_train[k]) for k in server_candidates])
            total_loss = sum([np.mean([loss for loss in loss_records[k]]) for k in server_candidates])
            
            g_values = {}
            for k in server_candidates:
                d_k = len(dict_users_train[k]) / total_data
                I_k = (np.mean(loss_records[k]) / total_loss) if total_loss !=0 else 0
                g_values[k] = (d_k + 0.5*I_k)
                
            server_users = max(g_values, key=g_values.get)
        else:
            server_users = server_candidates[rnd % len(server_candidates)]
        
        for snd in range(args.s_epochs):
            epoch += 1
            loss_locals = []
            loss_records = {k: [] for k in server_candidates}
            
            for idx in server_candidates:
                if idx != server_users:
                    print('server_users: {}, local_client: {}'.format(server_users, idx))
                    local = LocalUpdate(args, dataset_train, dict_users_train[idx], idx)
                    w, loss = local.train(copy.deepcopy(net_glob), current_lr)
                    local_weights[idx] = copy.deepcopy(w)
                    loss_records[idx].append(loss)
                    loss_locals.append(loss)
                    if loss <= best_loss[idx]:
                        best_loss[idx] = loss
                        torch.save(w, os.path.join(save_path,'{}_best_model.pkl'.format(idx)))
                        print('save the client {} best model'.format(idx))

            w_glob = FedAvg(local_weights)
            net_glob.load_state_dict(w_glob)
            
            print('server_users: {} training'.format(server_users))
            server = ServerUpdate(args, dataset_train, dict_users_train[server_users], server_users)
            w_server, loss_server = server.train(copy.deepcopy(net_glob).to(args.device), current_lr)
            net_glob.load_state_dict(w_server)
            loss_records[server_users].append(loss_server)
            loss_train_global.append(loss_server)
            if loss_server <= best_loss[server_users]:
                best_loss[server_users] = loss_server
                torch.save(w_server, os.path.join(save_path,'{}_best_model.pkl'.format(server_users)))
                print('save the client {} best model'.format(server_users))

            loss_avg = sum(loss_locals) / len(loss_locals)
            loss_train.append(loss_avg)
            print(f'Round {rnd+1} | Server: {server_users} | Local Loss: {loss_avg:.4f} | Server Loss: {loss_server:.4f}')
            
            current_lr *= 0.9
            print(f"New learning rate: {current_lr:.6f}")

    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('local train_loss')
    plt.savefig(os.path.join(save_path, 'local_{}_{}_{}_{}_non-iid.png'.format(args.dataset, args.model, args.r_epochs, args.num_users)))
    #
    plt.figure()
    plt.plot(range(len(loss_train_global)), loss_train_global)
    plt.ylabel('global train_loss')
    plt.savefig(os.path.join(save_path, 'global_{}_{}_{}_{}_non-iid.png'.format(args.dataset, args.model, args.r_epochs, args.num_users)))
    
    for i in range(args.num_users):
        print('Loading previously trained network: {}...'.format(i))
        # Load the model
        checkpoint = torch.load(os.path.join(save_path,'{}_best_model.pkl'.format(i)), map_location = lambda storage, loc: storage)
        model_dict = net_best_m[i].state_dict()
        checkpoint =  {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(checkpoint)
        net_best_m[i].load_state_dict(model_dict)
        del checkpoint
        torch.cuda.empty_cache()
        net_best_m[i].eval()
        # Test the model
        print('Testing client {}...'.format(i))
        client_data = DatasetSplit(dataset_test, dict_users_test[i])
        acc, loss = test_img(net_best_m[i], client_data, args)
        print(f'Client {i} - Best Accuracy: {acc:.2f}% | Loss: {loss:.4f}')