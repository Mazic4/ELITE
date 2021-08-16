import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from model import VAE,supervised_model
import utils
from utils import to_var
from data_loader import get_loader

import torch.nn as nn

import sys

method_sys = sys.argv[1]
normal_class = int(sys.argv[2])
data_sys = sys.argv[3]
ratio_outlier = float(sys.argv[4])
num_labels = int(sys.argv[5])
balance = False
if len(sys.argv) - 1 >= 6:
    num_out_class = int(sys.argv[6])
else:
    num_out_class = 9

print("Methods:", method_sys)
print("normal class:", normal_class)
print("data:", data_sys)
print("ratio_outlier:", ratio_outlier)
print("num_labels", num_labels)
print("balance", balance)
print("Num Outlier Classes", num_out_class)

hyperparameters = {
    'lr': 3e-4,
    'batch_size': 128,
    'num_iterations': 5000 // 128 * 150,
    'dataset': data_sys,
    'normal_class': normal_class,
    'n_val': num_labels // 2,
    'ratio_outlier': ratio_outlier,
    'balance': balance,
    'n_out_class': num_out_class
}

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
print("Set seed :", seed)

data_loader, visual_dataset = get_loader(hyperparameters['batch_size'],
                                         classes=hyperparameters['normal_class'],
                                         proportion=1 - hyperparameters['ratio_outlier'],
                                         mode="train",
                                         dataset=data_sys,
                                         n_val=hyperparameters['n_val'],
                                         balance=hyperparameters['balance'],
                                         n_out_class=hyperparameters['n_out_class']
                                         )
test_loader, test_dataset = get_loader(hyperparameters['batch_size'],
                                       classes=hyperparameters['normal_class'],
                                       proportion=0.90,
                                       n_items=10000,
                                       mode="test",
                                       dataset=hyperparameters['dataset'],
                                       balance=hyperparameters['balance']
                                       )

print(len(visual_dataset))
print(np.unique(visual_dataset.labels, return_counts=True))

print(len(test_dataset))
print(np.unique(test_dataset.labels, return_counts=True))

val_data = to_var(data_loader.dataset.data_val, requires_grad=False)
val_labels = to_var(data_loader.dataset.labels_val, requires_grad=False)


def build_model():
    net = supervised_model(dataset=data_sys)

    if torch.cuda.is_available():
        net.cuda()
        torch.backends.cudnn.benchmark = True

    opt = torch.optim.SGD(net.params(), lr=hyperparameters["lr"])

    return net, opt

def init_center_c(train_loader, net, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = torch.zeros(net.rep_dim).cuda()

    net.eval()
    with torch.no_grad():
        for _ in range(100):
            # get the inputs of the batch
            inputs, labels, idx = next(iter(data_loader))
            inputs = inputs.cuda()
            outputs = net(inputs)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c

def train_lre():
    net, opt = build_model()
    # method = "baseline_"
    method = method_sys

    smoothing_alpha = 0.9

    n_val = hyperparameters['n_val']

    temp_w = torch.zeros(5000 - n_val * 2).cuda()
    temp_cost = torch.zeros(5000 - n_val * 2)

    criterion = nn.CrossEntropyLoss()

    c = init_center_c(data_loader, net, eps=0.1)

    for i in tqdm(range(hyperparameters['num_iterations'])):
        net.train()

        # Line 2 get batch of data
        image, labels, idx = next(iter(data_loader))

        image = to_var(image, requires_grad=False)
        labels = to_var(labels, requires_grad=False)

        feats = net(image)

        eps_ = 1e-6

        dist = torch.sum((feats - c) ** 2, dim=1) ** 0.5
        feats_normal = (dist[labels == 1] + eps_)
        feats_abnormal = torch.clamp(10 - dist[labels == 0], min=0)

        loss = 0.5 * feats_normal.mean() + 0.5 * feats_abnormal.mean()

        # output = net(image)
        # loss = criterion(output, labels.long())

        opt.zero_grad()
        loss.backward()
        opt.step()

    return net,c

net,c = train_lre()
net.eval()

mask = np.zeros(len(test_dataset))
pred_prob_test_log = torch.zeros(len(test_dataset))
iter_test_loader = iter(test_loader)
for i in range(len(test_loader)):
    image, labels, idx = iter_test_loader.next()
    feats = net(image.cuda())
    dist = torch.sum((feats - c) ** 2, dim=1) ** 0.5
    pred_prob_test_log[idx] = dist.cpu().detach()
    # output = net(image.cuda())
    # pred_prob_test_log[idx] = output.cpu().detach()[:,0]
    mask[idx] = 1

print(np.sum(mask))

auc_log = []
auc_test_log = []

auc_test_log += [utils.eval_auc(pred_prob_test_log.cpu().detach().numpy(), test_dataset.labels)]

print("auc in training data loader:", auc_log)
print("auc in testing data loader:", auc_test_log)
