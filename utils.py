import torch
from torch.autograd import Variable
import torch.nn as nn
from sklearn import metrics
# import numpy as np


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def reconstruction_loss(x_reconstructed, x):
    raw_loss = nn.BCELoss(size_average=False)(x_reconstructed, x) / x.size(0)
    return raw_loss


def kl_divergence_loss(mean, logvar):
    return ((mean ** 2 + logvar.exp() - 1 - logvar) / 2)


def eval_auc(y_prob, y):
    # fpr, tpr, thresholds = metrics.roc_curve(y, y_prob, pos_label=2)
    score = metrics.roc_auc_score(y, y_prob)
    return 1 - score
