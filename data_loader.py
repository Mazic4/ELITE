import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
import numpy as np


class DatasetImbalanced():
    def __init__(self, n_items=5000, classes=9, proportion=0.9, n_val=5, random_seed=1, mode="train", dataset="cifar",
                 balance=False, n_out_class=9):

        if mode != "train":
            n_items = int(1e8)

        if dataset == "mnist":
            if mode == "train":
                self.dataset = datasets.MNIST('data', train=True, download=True)

            else:
                self.dataset = datasets.MNIST('data', train=False, download=True)
                proportion = 0.5
                n_val = 0

            self.transform = transforms.Compose([
                transforms.Resize([32, 32]),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif dataset == "cifar":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            if mode == "train":
                self.dataset = datasets.CIFAR10(root='./data', train=True, download=True)

            else:
                self.dataset = datasets.CIFAR10(root='./data', train=False, download=True)
                proportion = proportion
                # proportion = 0.5
                n_val = 0

        elif dataset == "svhn":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            if mode == "train":
                self.dataset = datasets.SVHN(root='./data', split='train', download=True)

            else:
                self.dataset = datasets.SVHN(root='./data', split='test', download=True)
                proportion = proportion
                n_val = 0

        elif dataset == "fmnist":
            self.transform = transforms.Compose([
                transforms.Resize([32, 32]),
                transforms.ToTensor()
            ])
            if mode == "train":
                self.dataset = datasets.FashionMNIST(root='./data', train=True, download=True)

            else:
                self.dataset = datasets.FashionMNIST(root='./data', train=False, download=True)
                proportion = proportion
                # proportion = 0.5
                n_val = 0

        self.dataset_name = dataset
        n_class = int(np.floor(n_items * proportion))

        self.data = []
        self.data_val = []
        self.labels = []
        self.labels_val = []

        label_source = None
        data_source = None
        if self.dataset_name == "mnist":
            data_source = self.dataset.data
            label_source = self.dataset.targets.clone().detach()
        elif self.dataset_name == "cifar":
            data_source = torch.from_numpy(self.dataset.data)
            label_source = torch.tensor(self.dataset.targets)
        elif self.dataset_name == "fmnist":
            data_source = self.dataset.data
            label_source = self.dataset.targets.clone().detach()
        elif self.dataset_name == "svhn":
            data_source = self.dataset.data
            label_source = torch.tensor(self.dataset.labels)

        if balance:
            n_val_outlier = int(2 * n_val * proportion)
            n_val_inlier = int(2 * n_val * (1 - proportion))
        else:
            n_val_outlier = n_val
            n_val_inlier = n_val

        # insert outliers
        tmp_idx = np.where(label_source != classes)[0]
        tmp_idx = torch.from_numpy(tmp_idx)
        img = data_source[tmp_idx[:(n_items - n_class) - n_val_outlier]]
        self.data.append(img if self.dataset_name != "svhn" else torch.from_numpy(img.transpose(0, 2, 3, 1)))

        cl = label_source[tmp_idx[:(n_items - n_class) - n_val_outlier]]
        self.labels.append((cl == classes).float())

        if mode == "train":
            # constrain labeled data to draw outliers from classes [0-n_out_class]
            out_classes = []
            class_id = 0
            while len(out_classes) < n_out_class:
                if class_id != classes:  # don't include inlier class
                    out_classes.append(class_id)
                class_id += 1
            tmp_idx = np.where(np.isin(label_source, out_classes))[0]

            img_val = data_source[tmp_idx[n_class - n_val_outlier:n_class]]
            for idx in range(n_val):
                if dataset == "mnist":
                    img_tmp = Image.fromarray(img_val[idx].numpy(), mode='L')
                    img_tmp = self.transform(img_tmp)
                    self.data_val.append(img_tmp.unsqueeze(0))
                elif dataset == "cifar":
                    img_tmp = self.transform(img_val[idx].numpy()).float()
                    self.data_val.append(img_tmp.unsqueeze(0))
                elif dataset == "svhn":
                    img_tmp = self.transform(img_val[idx].transpose([1, 2, 0])).float()
                    self.data_val.append(img_tmp.unsqueeze(0))
                elif dataset == "fmnist":
                    img_tmp = Image.fromarray(img_val[idx].numpy(), mode='L')
                    img_tmp = self.transform(img_tmp).float()
                    self.data_val.append(img_tmp.unsqueeze(0))

            cl_val = label_source[tmp_idx[n_class - n_val_outlier:n_class]]
            self.labels_val.append((cl_val == classes).float())

        # insert inliers
        tmp_idx = np.where(label_source == classes)[0]
        np.random.shuffle(tmp_idx)
        tmp_idx = torch.from_numpy(tmp_idx)
        img = data_source[tmp_idx[:n_class - n_val_inlier]]
        self.data.append(img if self.dataset_name != "svhn" else torch.from_numpy(img.transpose(0, 2, 3, 1)))

        cl = label_source[tmp_idx[:n_class - n_val_inlier]]
        self.labels.append((cl == classes).float())

        if mode == "train":
            img_val = data_source[tmp_idx[n_class - n_val_inlier:n_class]]
            for idx in range(img_val.shape[0]):
                if dataset == "mnist":
                    img_tmp = Image.fromarray(img_val[idx].numpy(), mode='L')
                    img_tmp = self.transform(img_tmp)
                    self.data_val.append(img_tmp.unsqueeze(0))
                elif dataset == "cifar":
                    img_tmp = self.transform(img_val[idx].numpy()).float()
                    self.data_val.append(img_tmp.unsqueeze(0))
                elif dataset == "svhn":
                    img_tmp = self.transform(img_val[idx].transpose([1, 2, 0])).float()
                    self.data_val.append(img_tmp.unsqueeze(0))
                elif dataset == "fmnist":
                    img_tmp = Image.fromarray(img_val[idx].numpy(), mode='L')
                    img_tmp = self.transform(img_tmp).float()
                    self.data_val.append(img_tmp.unsqueeze(0))

            cl_val = label_source[tmp_idx[n_class - n_val_inlier:n_class]]
            self.labels_val.append((cl_val == classes).float())

        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

        if mode == "train":
            self.data_val = torch.cat(self.data_val)
            self.labels_val = torch.cat(self.labels_val)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.dataset_name == "mnist":
            img = Image.fromarray(img.numpy(), mode='L')

            if self.transform is not None:
                img = self.transform(img)
        elif self.dataset_name == "cifar":
            img = self.transform(img.numpy()).float()
        elif self.dataset_name == "svhn":
            img = self.transform(img.numpy()).float()
        elif self.dataset_name == "fmnist":
            img = Image.fromarray(img.numpy(), mode='L')
            img = self.transform(img).float()

        return img, target, index


def get_loader(batch_size, classes=9, n_items=5000, proportion=0.9, n_val=50, mode='train', dataset="cifar",
               balance=False, n_out_class=9):
    """Build and return data loader."""

    dataset = DatasetImbalanced(classes=classes, n_items=n_items, proportion=proportion,
                                n_val=n_val, mode=mode, dataset=dataset, balance=False, n_out_class=n_out_class)

    shuffle = False
    if mode == 'train':
        shuffle = True
    shuffle = True
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             pin_memory=True
                             )
    return data_loader, dataset
