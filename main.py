import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from model import VAE
import utils
from utils import to_var
from data_loader import get_loader
# import IPython
# import gc
# import matplotlib
# import torchvision
# import torch.nn as nn
# from data_loader import *
# import torch.nn.functional as F
# from torch.autograd import Variable

import sys

method_sys = sys.argv[1]
normal_class = int(sys.argv[2])
data_sys = sys.argv[3]
ratio_outlier = float(sys.argv[4])
num_labels = int(sys.argv[5])
balance = False
meta_scale = int(sys.argv[6])
margin = int(sys.argv[7])
num_out_class = 9

print("Methods:", method_sys)
print("normal class:", normal_class)
print("data:", data_sys)
print("ratio_outlier:", ratio_outlier)
print("num_labels", num_labels)
print("balance", balance)
print("Num Outlier Classes", num_out_class)
print ("meta_scale", meta_scale)
print ("margin", margin)

hyperparameters = {
    'lr': 3e-4,
    'batch_size': 128,
    'num_iterations': 5000 // 128 * 150,
    # 'num_iterations':500,
    'dataset': data_sys,
    'normal_class': normal_class,
    'n_val': num_labels // 2,
    'ratio_outlier': ratio_outlier,
    'balance': balance,
    'n_out_class': num_out_class,
    'meta_scale': meta_scale,
    'margin': margin
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
    net = VAE(dataset=data_sys)

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


def init_margin(model, images, c):
    model.train()
    images = utils.to_var(images, requires_grad=False)
    feats = model(images)
    dist = torch.sum((feats - c) ** 2, dim=1) ** 0.5

    return torch.max(torch.sum((feats - c) ** 2, dim=1) ** 0.5)


def train_lre():
    net, opt = build_model()

    method = method_sys

    smoothing_alpha = 0.9

    n_val = hyperparameters['n_val']

    temp_w = torch.zeros(5000 - n_val * 2).cuda()
    temp_l = torch.zeros(5000 - n_val * 2).cuda()
    temp_cost = torch.zeros(5000 - n_val * 2)
    inliers_temp = []
    outliers_temp = []
    inlier_scores_log = []
    outlier_scores_log = []
    accuracy_log = []
    labeled_accuracy_log = []

    c = init_center_c(data_loader, net, eps=0.1)
    margin = init_margin(net, val_data, c).item()
    print ('margin:', margin)

    for i in tqdm(range(hyperparameters['num_iterations'])):
        net.train()

        image, labels, idx = next(iter(data_loader))

        image = to_var(image, requires_grad=False)
        labels = to_var(labels, requires_grad=False)

        eps_ = 1e-6
        meta_net = VAE(dataset=data_sys)
        meta_net.load_state_dict(net.state_dict())

        if torch.cuda.is_available():
            meta_net.cuda()

        feats = meta_net(image)

        loss_ = torch.sum((feats - c)**2, dim=1)**0.5

        eps = to_var(torch.zeros(len(loss_)))

        scores = loss_ * eps

        loss_meta = torch.sum(scores)

        meta_net.zero_grad()

        grads = torch.autograd.grad(loss_meta, (meta_net.params()), create_graph=True)

        meta_net.update_params(hyperparameters['lr'] * hyperparameters["meta_scale"], source_params=grads)

        val_feats = meta_net(val_data)

        dist = torch.sum((val_feats - c) ** 2, dim=1) ** 0.5
        val_feats_normal = (dist[val_labels == 1] + eps_)
        val_feats_abnormal = torch.clamp(hyperparameters["margin"] - dist[val_labels == 0], min=0)

        meta_loss = 0.5 * val_feats_normal.mean() + 0.5 * val_feats_abnormal.mean()
        grad_eps = torch.autograd.grad(meta_loss, eps, only_inputs=True)[0]

        w = -grad_eps

        feats = net(image)
        loss = torch.sum((feats - c) ** 2, dim=1) ** 0.5

        l_f = (loss * w.cuda()).mean()
        l_f_log = w
        # print ([i for i in zip(labels.cpu().detach().numpy(), l_f_log.cpu().detach().numpy())])

        temp_w[idx] += l_f_log
        temp_l[idx] = labels

        if i % 1000 == 0:
            print(dist[val_labels == 1].mean(), dist[val_labels == 0].mean())
            print(loss.mean())
            print(l_f.mean())

        # temp_cost[idx] = smoothing_alpha * temp_cost[idx] + (1 - smoothing_alpha) * loss.cpu().detach()
        temp_cost[idx] = loss.cpu().detach()

        metric = utils.eval_auc(temp_cost, visual_dataset.labels)
        # log_training_loss = torch.where(visual_dataset.labels > 0, temp_cost, margin - temp_cost)
        log_training_loss = meta_loss.item()
        accuracy_log += [metric]
        labeled_accuracy_log += [log_training_loss]

        inliers_temp += [np.mean(loss.cpu().detach().numpy()[labels.cpu().detach().numpy() == 1])]
        if np.sum([labels.cpu().detach().numpy() == 0]) > 1:
            outliers_temp += [np.mean(loss.cpu().detach().numpy()[labels.cpu().detach().numpy() == 0])]

        if len(inliers_temp) == 100:
            inlier_scores_log += [np.mean(inliers_temp)]
            outlier_scores_log += [np.mean(outliers_temp)]
            inliers_temp = []
            outliers_temp = []

        log_iter = hyperparameters["num_iterations"]-1

        opt.zero_grad()
        l_f.backward()
        opt.step()

        if i % 1000 == 0:

            pred_outliers_idx = torch.argsort(temp_w)[:500]
            print(torch.sum(visual_dataset.labels[pred_outliers_idx] == 0))

            pred_outliers_idx_1 = torch.argsort(temp_cost)[-500:]
            print(torch.sum(visual_dataset.labels[pred_outliers_idx_1] == 0))

            print(utils.eval_auc(temp_cost, visual_dataset.labels))

        # if i % plot_step == 0:
        #     net.eval()
        #
        #     # IPython.display.clear_output()
        #     fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        #     ax1, ax2 = axes.ravel()
        #
        #     ax1.plot(inlier_scores_log, label='inlier_loss')
        #     ax1.plot(outlier_scores_log, label='outlier_loss')
        #     ax1.set_ylabel("Losses")
        #     ax1.set_xlabel("Iteration")
        #     ax1.legend()
        #
        #     plt.show()

        # return accuracy
    return net, c


num_repeats = 5
# proportions = [0.9, 0.95, 0.98, 0.99, 0.995]
proportions = [0.9]
pred_prob_log = defaultdict(list)
true_label_log = defaultdict(list)

pred_prob_test_log = defaultdict(list)
true_label_test_log = defaultdict(list)

for prop in proportions:
    net, c = train_lre()
    net.eval()

    mask = np.zeros(5000 - hyperparameters['n_val'] * 2)
    pred_prob_log[prop] = torch.zeros(5000 - hyperparameters['n_val'] * 2)
    iter_data_loader = iter(data_loader)
    for i in range(len(data_loader)):
        image, labels, idx = iter_data_loader.next()
        feats = net(image.cuda())
        unl_dist = torch.sum((feats - c) ** 2, dim=1) ** 0.5
        pred_prob_log[prop][idx] = unl_dist.cpu().detach()
        mask[idx] = 1

    print(np.sum(mask))

    mask = np.zeros(len(test_dataset))
    pred_prob_test_log[prop] = torch.zeros(len(test_dataset))
    iter_test_loader = iter(test_loader)
    for i in range(len(test_loader)):
        image, labels, idx = iter_test_loader.next()
        feats = net(image.cuda())
        unl_dist = torch.sum((feats - c) ** 2, dim=1) ** 0.5
        pred_prob_test_log[prop][idx] = unl_dist.cpu().detach()
        mask[idx] = 1

    print(np.sum(mask))

auc_log = []
auc_test_log = []
plt.figure()
for prop in proportions:
    auc_log += [utils.eval_auc(pred_prob_log[prop].cpu().detach().numpy(), visual_dataset.labels)]

for prop in proportions:
    auc_test_log += [utils.eval_auc(pred_prob_test_log[prop].cpu().detach().numpy(), test_dataset.labels)]

plt.bar(np.arange(len(proportions)), auc_log)
print("auc in training data loader:", auc_log)
print("auc in testing data loader:", auc_test_log)
