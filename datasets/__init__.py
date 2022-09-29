import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset
from torchvision import transforms
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def build_al_dataset(args):
    if args.dataset == 'MNIST':
        train_ds = build_mnist('train')
        test_ds = build_mnist('test')
        al_ds = build_mnist('al')
        n_classes = 10
        return train_ds, test_ds, al_ds, n_classes
    elif args.dataset == 'LETTER':
        train_ds = build_tabular_data(6, 'train')
        test_ds = build_tabular_data(6, 'test')
        al_ds = build_tabular_data(6, 'al')
        n_classes = 26
        return train_ds, test_ds, al_ds, n_classes
    elif args.dataset == 'PENDIGITS':
        train_ds = build_tabular_data(32, 'train')
        test_ds = build_tabular_data(32, 'test')
        al_ds = build_tabular_data(32, 'al')
        n_classes = 10
        return train_ds, test_ds, al_ds, n_classes
    elif args.dataset == '2D':
        train_ds = build_tabular_data('train')
        test_ds = build_tabular_data('test')
        al_ds = build_tabular_data('al')
        n_classes = 2
        return train_ds, test_ds, al_ds, n_classes
    elif args.dataset == 'FashionMNIST':
        train_ds = build_fashion_mnist('train')
        test_ds = build_fashion_mnist('test')
        al_ds = build_fashion_mnist('al')
        n_classes = 10
    elif args.dataset == 'CIFAR10':
        train_ds = build_cifar10('train')
        test_ds = build_cifar10('test')
        al_ds = build_cifar10('al')
        n_classes = 10
    elif args.dataset == 'SVHN':
        train_ds = build_svhn('train')
        test_ds = build_svhn('test')
        al_ds = build_svhn('al')
        n_classes = 10
    elif args.dataset == 'CIFAR10':
        train_ds = build_cifar10('train')
        test_ds = build_cifar10('test')
        al_ds = build_cifar10('al')
        n_classes = 10
    else:
        raise ValueError(f"Dataset {args.dataset} not implemented.")
    return train_ds, test_ds, al_ds, n_classes


def build_dataset(args):
    if args.dataset == 'MNIST':
        train_ds = build_mnist('train')
        test_ds = build_mnist('test')
        n_classes = 10
        return train_ds, test_ds, n_classes
    elif args.dataset == 'FashionMNIST':
        train_ds = build_fashion_mnist('train')
        test_ds = build_fashion_mnist('test')
        n_classes = 10
        return train_ds, test_ds, n_classes
    elif args.dataset == 'CIFAR10':
        train_ds = build_cifar10('train')
        test_ds = build_cifar10('test')
        n_classes = 10
        return train_ds, test_ds, n_classes
    elif args.dataset == 'SVHN':
        train_ds = build_svhn('train')
        test_ds = build_svhn('test')
        n_classes = 10
        return train_ds, test_ds, n_classes
    elif args.dataset == "LETTER":
        train_ds = build_tabular_data(6, 'train')
        test_ds = build_tabular_data(6, 'test')
        n_classes = 26
    elif args.dataset == "PENDIGITS":
        train_ds = build_tabular_data(32, 'train')
        test_ds = build_tabular_data(32, 'test')
        n_classes = 10
    elif args.dataset == "2D":
        train_ds = build_2d_dataset('train')
        test_ds = build_2d_dataset('test')
        n_classes = 2
    else:
        raise ValueError(f"Dataset {args.dataset} not implemented.")
    return train_ds, test_ds, n_classes


def build_ood_dataset(args):
    ood_datasets = {}
    if 'SVHN' in args.datasets_ood:
        ood_datasets['SVHN'] = build_svhn('test')
    if 'CIFAR100' in args.datasets_ood:
        ood_datasets['CIFAR100'] = build_cifar100('test')
    if 'FashionMNIST' in args.datasets_ood:
        ood_datasets['FashionMNIST'] = build_fashion_mnist('test')
    if 'MNIST' in args.datasets_ood:
        ood_datasets['MNIST'] = build_mnist('test')
    if 'GAUSSIAN_NOISE' in args.datasets_ood:
        ood_datasets['IMG_GAUSSIAN_NOISE'] = TensorDataset(torch.randn(10000, 3, 32, 32), torch.empty(10000))
    if 'LETTER' in args.datasets_ood:
        ood_datasets['LETTER'] = build_tabular_data(6, 'test')
    if 'PENDIGITS' in args.datasets_ood:
        ood_datasets['PENDIGITS'] = build_tabular_data(32, 'test')
    if 'TAB_GAUSSIAN_NOISE' in args.datasets_ood:
        ood_datasets['TAB_GAUSSIAN_NOISE'] = TensorDataset(torch.randn(10000, 16), torch.empty(10000))
    return ood_datasets


def build_mnist(split, path='data/'):
    mean, std = (0.1307,), (0.3081,)
    test_transform = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean*3, std*3),
    ])
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize(size=(32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean * 3, std * 3),
    ])
    if split == 'train':
        ds = torchvision.datasets.MNIST(path, train=True, download=True, transform=train_transform)
    elif split == 'test':
        ds = torchvision.datasets.MNIST(path, train=False, download=True, transform=test_transform)
    elif split == "al":
        ds = torchvision.datasets.MNIST(path, train=True, download=True, transform=test_transform)
    return ds


def build_fashion_mnist(split, path='data/'):
    mean, std = (0.1307,), (0.3081,)
    test_transform = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean*3, std*3),
    ])
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize(size=(32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean * 3, std * 3),
    ])
    if split == 'train':
        ds = torchvision.datasets.FashionMNIST(path, train=True, download=True, transform=train_transform)
    elif split == 'test':
        ds = torchvision.datasets.FashionMNIST(path, train=False, download=True, transform=test_transform)
    elif split == "al":
        ds = torchvision.datasets.FashionMNIST(path, train=True, download=True, transform=test_transform)
    return ds


def build_cifar10(split, path='data/'):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if split == 'train':
        ds = torchvision.datasets.CIFAR10(path, train=True, download=True, transform=train_transform)
    elif split == 'test':
        ds = torchvision.datasets.CIFAR10(path, train=False, download=True, transform=eval_transform)
    elif split == "al":
        ds = torchvision.datasets.CIFAR10(path, train=True, download=True, transform=eval_transform)
    return ds


def build_cifar100(split, path='data/'):
    mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if split == 'train':
        ds = torchvision.datasets.CIFAR100(path, train=True, download=True, transform=train_transform)
    elif split == 'test':
        ds = torchvision.datasets.CIFAR100(path, train=False, download=True, transform=eval_transform)
    elif split == 'al':
        ds = torchvision.datasets.CIFAR100(path, train=True, download=True, transform=eval_transform)
    return ds


def build_svhn(split, path='data/'):
    mean, std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if split == 'train':
        ds = torchvision.datasets.SVHN(path, split='train', download=True, transform=train_transform)
    elif split == 'test':
        ds = torchvision.datasets.SVHN(path, split='test', download=True, transform=eval_transform)
    elif split == "al":
        ds = torchvision.datasets.SVHN(path, split='train', download=True, transform=eval_transform)
    return ds


def build_tabular_data(data_id, split, path='data/'):
    X, y = fetch_openml(data_id=data_id, data_home=path, return_X_y=True)
    X = X.values
    y = LabelEncoder().fit_transform(y.values)
    train, test = train_test_split(np.arange(len(X)), random_state=0, test_size=0.25)
    scaler = StandardScaler().fit(X[train])
    if split in ["train", "al"]:
        X_train = torch.from_numpy(scaler.transform(X[train])).float()
        y_train = torch.from_numpy(y[train]).long()
        return TensorDataset(X_train, y_train)
    elif split == "test":
        X_test = torch.from_numpy(scaler.transform(X[test])).float()
        y_test = torch.from_numpy(y[test]).long()
        return TensorDataset(X_test, y_test)

def build_2d_dataset(split):
    X, y = make_moons(n_samples=500, noise=0.2, random_state=1)
    y %= 2
    X -= np.mean(X, axis=0, keepdims=True)
    X /= np.std(X, axis=0, keepdims=True)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    n_samples_new = 10
    noise = 0.2

    X = torch.cat([
        X,
        torch.randn((n_samples_new, 2)) * noise + 3,
        torch.randn((n_samples_new, 2)) * noise - 3,
        torch.randn((n_samples_new, 2)) * noise - torch.Tensor([3, -3]),
        torch.randn((n_samples_new, 2)) * noise + torch.Tensor([3, -3]),
        torch.randn((n_samples_new, 2)) * noise + torch.Tensor([0, -3]),
        torch.randn((n_samples_new, 2)) * noise + torch.Tensor([0, 3]),
        torch.randn((n_samples_new, 2)) * noise + torch.Tensor([-3, 0]),
        torch.randn((n_samples_new, 2)) * noise + torch.Tensor([3, 0]),
    ])
    y = torch.cat([
        y,
        torch.ones(n_samples_new).long() * 1,
        torch.ones(n_samples_new).long() * 1,
        torch.ones(n_samples_new).long() * 0,
        torch.ones(n_samples_new).long() * 0,
        torch.ones(n_samples_new).long() * 1,
        torch.ones(n_samples_new).long() * 0,
        torch.ones(n_samples_new).long() * 1,
        torch.ones(n_samples_new).long() * 0,
    ])

    train, test = train_test_split(np.arange(len(X)), random_state=0, test_size=0.25)
    if split in ["train", "al"]:
        return TensorDataset(X[train], y[train])
    elif split == "test":
        return TensorDataset(X[test], y[test])