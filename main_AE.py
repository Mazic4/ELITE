import os
import sys
import random
import collections
import numpy as np
import torch
# import torchvision
import utils
from tqdm import tqdm
from model import AE
from data_loader import get_loader
# import matplotlib.pyplot as plt
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# import IPython
# import gc
# import matplotlib


def build_model(hparams):
    net = AE(dataset=hparams['dataset'])
    if torch.cuda.is_available():
        net.cuda()
        torch.backends.cudnn.benchmark = True
    opt = torch.optim.SGD(net.params(), lr=hparams["lr"])
    return net, opt

def init_margin(model, images):
    images = utils.to_var(images, requires_grad=False)
    reconstructed_x = model(images)
    return torch.max(torch.sum((reconstructed_x - images)**2, dim=[1, 2, 3])**0.5)


def train_lre(hparams, data_loader, visual_dataset, val_data, val_labels):
    net, opt = build_model(hparams)
    
    margin = init_margin(net, val_data).cpu().item()
    print ("margin:", margin)
    
    method = hyperparameters['method']
    
    #net, opt = build_model(hparams)

    # plot_step = 10000000
    smoothing_alpha = 0.9
    n_val = hparams['n_val']

    temp_w = torch.zeros(5000 - n_val * 2).cuda()
    temp_cost = torch.zeros(5000 - n_val * 2)
    inliers_temp = []
    outliers_temp = []
    inlier_scores_log = []
    outlier_scores_log = []

    for i in tqdm(range(hparams['num_iterations'])):
        net.train()
        image, labels, idx = next(iter(data_loader))

        image = utils.to_var(image, requires_grad=False)
        labels = utils.to_var(labels, requires_grad=False)

        eps_ = 1e-6
        meta_net = AE(dataset=hparams['dataset'])
        meta_net.load_state_dict(net.state_dict())

        if torch.cuda.is_available():
            meta_net.cuda()

        reconstructed_x = meta_net(image)
        loss_ = torch.sum((reconstructed_x - image)**2, dim=[1, 2, 3])**0.5
        eps = utils.to_var(torch.zeros(len(loss_)))
        scores = loss_ * eps
        loss_meta = torch.sum(scores)

        meta_net.zero_grad()
        grads = torch.autograd.grad(loss_meta, (meta_net.params()), create_graph=True)

        if data_sys == "cifar":
            meta_scale = int(n_val)
        else:
            # meta_scale = int(n_val // 10)
            meta_scale = 5

        meta_net.update_params(meta_scale * hparams['mlr'], source_params=grads)
        reconstructed_val = meta_net(val_data)

        dist = torch.sum((reconstructed_val - val_data) ** 2, dim=[1, 2, 3]) ** 0.5
        val_feats_normal = (dist[val_labels == 1] + eps_)
        val_feats_abnormal = torch.clamp(margin - dist[val_labels == 0], min=0)

        meta_loss = 0.5 * val_feats_normal.mean() + 0.5 * val_feats_abnormal.mean()
        grad_eps = torch.autograd.grad(meta_loss, eps, only_inputs=True)[0]

        w = -grad_eps
        temp_w[idx] -= grad_eps

        reconstructed_x = net(image)
        loss = torch.sum((reconstructed_x - image) ** 2, dim=[1, 2, 3]) ** 0.5
        l_f = (loss * w.cuda()).mean()

        if i % 1000 == 0:
            print(dist[val_labels == 1].mean(),
                  dist[val_labels == 0].mean())
            print(loss.mean())
            print(l_f.mean())

        temp_cost[idx] = smoothing_alpha * temp_cost[idx] + (1 - smoothing_alpha) * loss.cpu().detach()

        inliers_temp += [np.mean(loss.cpu().detach().numpy()
                                 [labels.cpu().detach().numpy() == 1])]
        if np.sum([labels.cpu().detach().numpy() == 0]) > 1:
            outliers_temp += [np.mean(loss.cpu().detach().numpy()
                                      [labels.cpu().detach().numpy() == 0])]

        if len(inliers_temp) == 100:
            inlier_scores_log += [np.mean(inliers_temp)]
            outlier_scores_log += [np.mean(outliers_temp)]
            inliers_temp = []
            outliers_temp = []

    return net


def run_training(hparams):
    data_loader, visual_dataset = get_loader(hparams['batch_size'],
                                             classes=hparams['normal_class'],
                                             proportion=1 - hparams['ratio_outlier'],
                                             mode="train",
                                             dataset=hparams['dataset'],
                                             n_val=hparams['n_val'],
                                             n_out_class=hparams['n_out_class']
                                             )
    test_loader, test_dataset = get_loader(hparams['batch_size'],
                                           classes=hparams['normal_class'],
                                           proportion=0.90,
                                           n_items=10000,
                                           mode="test",
                                           dataset=hparams['dataset']
                                           )

    # Dataset information
    print("Training Dataset")
    print("  Num Samples: ", len(visual_dataset))
    train_occurences = np.unique(visual_dataset.labels, return_counts=True)
    print("  | Label | Frequency |")
    print("  |-------|-----------|")
    for label, frequency in zip(train_occurences[0], train_occurences[1]):
        print("  | ", label, " | %9d |" % frequency)

    print("Testing Dataset")
    print("  Num Samples: ", len(test_dataset))
    train_occurences = np.unique(test_dataset.labels, return_counts=True)
    print("  | Label | Frequency |")
    print("  |-------|-----------|")
    for label, frequency in zip(train_occurences[0], train_occurences[1]):
        print("  | ", label, " | %9d |" % frequency)

    val_data = utils.to_var(data_loader.dataset.data_val, requires_grad=False)
    val_labels = utils.to_var(data_loader.dataset.labels_val, requires_grad=False)

    pred_prob_log = collections.defaultdict(list)
    pred_prob_test_log = collections.defaultdict(list)

    net = train_lre(hparams, data_loader, visual_dataset, val_data, val_labels)
    net.eval()

    mask = np.zeros(5000 - hparams['n_val'] * 2)
    pred_prob_log[hparams['ratio_outlier']] = torch.zeros(5000 - hparams['n_val'] * 2)
    iter_data_loader = iter(data_loader)
    for i in range(len(data_loader)):
        image, labels, idx = iter_data_loader.next()
        reconstructed_x = net(image.cuda())
        unl_dist = torch.sum((reconstructed_x - image.cuda()) ** 2, dim=[1, 2, 3]) ** 0.5
        pred_prob_log[hparams['ratio_outlier']][idx] = unl_dist.cpu().detach()
        mask[idx] = 1

    print(np.sum(mask))

    mask = np.zeros(len(test_dataset))
    pred_prob_test_log[hparams['ratio_outlier']] = torch.zeros(len(test_dataset))
    iter_test_loader = iter(test_loader)
    for i in range(len(test_loader)):
        image, labels, idx = iter_test_loader.next()
        reconstructed_x = net(image.cuda())
        unl_dist = torch.sum((reconstructed_x - image.cuda()) ** 2, dim=[1, 2, 3]) ** 0.5
        pred_prob_test_log[hparams['ratio_outlier']][idx] = unl_dist.cpu().detach()
        mask[idx] = 1

    print(np.sum(mask))

    auc_train = utils.eval_auc(pred_prob_log[hparams['ratio_outlier']].cpu().detach().numpy(), visual_dataset.labels)
    auc_test = utils.eval_auc(pred_prob_test_log[hparams['ratio_outlier']].cpu().detach().numpy(), test_dataset.labels)

    print("auc in training data loader:", auc_train)
    print("auc in testing data loader:", auc_test)
    log.write(str(auc_train))
    log.write(", ")
    log.write(str(auc_test))
    log.write("\n")


def set_rng_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_log(hparams):
    global log_file
    if log_file is None:
        log_file = "{dir}/results_{alg}_{data}_{its:d}.csv".format(
            dir=out_dir,
            alg=hparams['method'],
            data=hparams['dataset'],
            its=hparams['num_iterations']
        )

    append_to_log = os.path.exists(log_file)
    log = open(log_file, mode="a")
    # if the log is new, output header row
    if not append_to_log:
        for parameter in hyperparameters:
            log.write(parameter)
            log.write(", ")
        for i in range(0, hparams['num_iterations'], 1000):
            log.write("AUC ")
            log.write(str(i))
            log.write("th it")
            log.write(", ")
        log.write("Train AUC, ")
        log.write("Test AUC, \n")
    # log all hyperparameters
    for value in hyperparameters.values():
        log.write(str(value))
        log.write(", ")

    return log


if __name__ == "__main__":
    # Read hyperparameters from command line arguments
    method_sys = sys.argv[1]            # algorithm to use: baseline/ours
    data_sys = sys.argv[2]              # datset to use mnist/cifar
    normal_class = int(sys.argv[3])     # integer ID of inlier class 0-10
    ratio_outlier = float(sys.argv[4])  # ratio of outliers in the training set 0.0-1.0
    num_labels = int(sys.argv[5])       # number of labeled datapoints in the training dataset
    if len(sys.argv) - 1 >= 6:
        log_file = sys.argv[6]
        out_dir = os.path.dirname(log_file)
        if out_dir == "":
            out_dir = "ELITE/"
    else:
        log_file = None
        out_dir = "./results-AE"
    if len(sys.argv) - 1 >= 7:
        num_out_class = int(sys.argv[7])
    else:
        num_out_class = 9

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    hyperparameters = collections.OrderedDict([
        ('method', method_sys),
        ('dataset', data_sys),
        ('normal_class', normal_class),
        ('ratio_outlier', ratio_outlier),
        ('n_val', num_labels // 2),
        ('lr', 0.0006), 
        # Learning rate for network parameters of autoencoder
        ('mlr', 0.0006),      # Meta learning rate for w
        ('batch_size', 128),
        ('n_out_class', num_out_class),
        ('num_iterations', 5000 // 128 * 150)
    ])

    print("Methods:", hyperparameters['method'])
    print("normal class:", hyperparameters['normal_class'])
    print("data:", hyperparameters['normal_class'])
    print("ratio_outlier:", hyperparameters['ratio_outlier'])
    print("num_labels", hyperparameters['n_val'] * 2)

    # Set rng seeds to fixed value for repeatability
    seed = 1
    set_rng_seeds(seed)
    print("Set seed :", seed)

    log = get_log(hyperparameters)
    run_training(hyperparameters)
    log.close()
