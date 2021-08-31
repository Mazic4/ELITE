import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from model import SVDD
import utils
from utils import to_var
from data_loader import get_loader


class Trainer():
    def __init__(self, args):
        self.args = args

        self.get_data()
        self.build_model()

        pass

    def get_data(self):

        hyperparameters = self.args

        self.data_loader, self.train_dataset = get_loader(hyperparameters['batch_size'],
                                                 classes=hyperparameters['normal_class'],
                                                 proportion=1 - hyperparameters['ratio_outlier'],
                                                 mode="train",
                                                 dataset=data_sys,
                                                 n_val=hyperparameters['n_val'],
                                                 balance=hyperparameters['balance'],
                                                 n_out_class=hyperparameters['n_out_class']
                                                 )
        self.test_loader, self.test_dataset = get_loader(hyperparameters['batch_size'],
                                               classes=hyperparameters['normal_class'],
                                               proportion=0.90,
                                               n_items=10000,
                                               mode="test",
                                               dataset=hyperparameters['dataset'],
                                               balance=hyperparameters['balance']
                                               )

        self.val_data = to_var(self.data_loader.dataset.data_val, requires_grad=False)
        self.val_labels = to_var(self.data_loader.dataset.labels_val, requires_grad=False)

        print ("-------------------Data Info--------------------------")
        print(len(self.train_dataset))
        print(np.unique(self.train_dataset.labels, return_counts=True))

        print(len(self.test_dataset))
        print(np.unique(self.test_dataset.labels, return_counts=True))



    def build_model(self):
        self.net = SVDD(dataset=data_sys)

        if torch.cuda.is_available():
            self.net.cuda()
            torch.backends.cudnn.benchmark = True

        self.opt = torch.optim.SGD(self.net.params(), lr=self.args["lr"])


    def init_center_c(self, eps=0.1):
        """
        Initialize hypersphere center c as the mean from an initial forward pass on the data.
        """
        n_samples = 0
        c = torch.zeros(self.net.rep_dim).cuda()

        self.net.eval()
        with torch.no_grad():
            for _ in range(100):
                # get the inputs of the batch
                inputs, labels, idx = next(iter(self.data_loader))
                inputs = inputs.cuda()
                outputs = self.net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


    def init_margin(self, images, c):
        self.net.train()
        images = utils.to_var(images, requires_grad=False)
        feats = self.net(images)

        return torch.max(torch.sum((feats - c) ** 2, dim=1) ** 0.5)


    def train_lre(self):
        net, opt = self.net, self.opt
        hyperparameters = self.args

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

        self.c = self.init_center_c(eps=0.1)
        margin = self.init_margin(self.val_data, self.c).item()
        print ('margin:', margin)

        for i in tqdm(range(hyperparameters['num_iterations'])):
            net.train()

            image, labels, idx = next(iter(self.data_loader))

            image = to_var(image, requires_grad=False)
            labels = to_var(labels, requires_grad=False)

            eps_ = 1e-6
            meta_net = SVDD(dataset=data_sys)
            meta_net.load_state_dict(self.net.state_dict())

            if torch.cuda.is_available():
                meta_net.cuda()

            feats = meta_net(image)

            loss_ = torch.sum((feats - self.c)**2, dim=1)**0.5

            eps = to_var(torch.zeros(len(loss_)))

            scores = loss_ * eps

            loss_meta = torch.sum(scores)

            meta_net.zero_grad()

            grads = torch.autograd.grad(loss_meta, (meta_net.params()), create_graph=True)

            meta_net.update_params(hyperparameters['lr'] * hyperparameters["meta_scale"], source_params=grads)

            val_feats = meta_net(self.val_data)

            dist = torch.sum((val_feats - self.c) ** 2, dim=1) ** 0.5
            val_feats_normal = (dist[self.val_labels == 1] + eps_)
            val_feats_abnormal = torch.clamp(margin - dist[self.val_labels == 0], min=0)

            meta_loss = 0.5 * val_feats_normal.mean() + 0.5 * val_feats_abnormal.mean()
            grad_eps = torch.autograd.grad(meta_loss, eps, only_inputs=True)[0]

            w = -grad_eps

            feats = net(image)
            loss = torch.sum((feats - self.c) ** 2, dim=1) ** 0.5

            l_f = (loss * w.cuda()).mean()
            l_f_log = w

            temp_w[idx] += l_f_log
            temp_l[idx] = labels

            temp_cost[idx] = loss.cpu().detach()

            metric = utils.eval_auc(temp_cost, self.train_dataset.labels)
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

            opt.zero_grad()
            l_f.backward()
            opt.step()

            if i % 1000 == 0:
                print ("\n")
                print("The AUC score of train data is", utils.eval_auc(temp_cost, self.train_dataset.labels))


    def main(self):
        self.train_lre()
        self.eval()

    def eval(self):

        hyperparameters = self.args

        self.net.eval()

        pred_prob_log = torch.zeros(5000 - hyperparameters['n_val'] * 2)
        iter_data_loader = iter(self.data_loader)
        for i in range(len(self.data_loader)):
            image, labels, idx = iter_data_loader.next()
            feats = self.net(image.cuda())
            unl_dist = torch.sum((feats - self.c) ** 2, dim=1) ** 0.5
            pred_prob_log[idx] = unl_dist.cpu().detach()


        pred_prob_test_log = torch.zeros(len(self.test_dataset))
        iter_test_loader = iter(self.test_loader)
        for i in range(len(self.test_loader)):
            image, labels, idx = iter_test_loader.next()
            feats = self.net(image.cuda())
            unl_dist = torch.sum((feats - self.c) ** 2, dim=1) ** 0.5
            pred_prob_test_log[idx] = unl_dist.cpu().detach()

        auc_log = utils.eval_auc(pred_prob_log.cpu().detach().numpy(), self.train_dataset.labels)
        auc_test_log = utils.eval_auc(pred_prob_test_log.cpu().detach().numpy(), self.test_dataset.labels)

        print("auc in training data loader:", auc_log)
        print("auc in testing data loader:", auc_test_log)




if __name__ == "__main__":
    import sys

    method_sys = sys.argv[1]
    normal_class = int(sys.argv[2])
    data_sys = sys.argv[3]
    ratio_outlier = float(sys.argv[4])
    num_labels = int(sys.argv[5])
    balance = False
    meta_scale = int(sys.argv[6])
    num_out_class = 9

    print("Methods:", method_sys)
    print("normal class:", normal_class)
    print("data:", data_sys)
    print("ratio_outlier:", ratio_outlier)
    print("num_labels", num_labels)
    print("balance", balance)
    print("Num Outlier Classes", num_out_class)
    print ("meta_scale", meta_scale)

    hyperparameters = {
        'lr': 3e-4,
        'batch_size': 128,
        'num_iterations': 5000 // 128 * 150,
        'dataset': data_sys,
        'normal_class': normal_class,
        'n_val': num_labels // 2,
        'ratio_outlier': ratio_outlier,
        'balance': balance,
        'n_out_class': num_out_class,
        'meta_scale': meta_scale,
    }

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print("Set seed :", seed)

    trainer = Trainer(hyperparameters)
    trainer.main()
