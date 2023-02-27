# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..  
# *************************************************************************

# Code ported from https://github.com/QinbinLi/MOON/blob/main/main.py

import copy
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from .fedavg import FedAvg
import torch.nn.functional as F

import sys
sys.path.insert(0, '../')
from utils import compute_accuracy
from loss import FedDecorrLoss


class ContrastiveModelWrapper(nn.Module):

    def __init__(self, base_model, use_proj_head, proj_dim):
        super(ContrastiveModelWrapper, self).__init__()
        self.features = base_model
        self.repres_dim = base_model.classifier.in_features
        self.n_class = base_model.classifier.out_features
        self.use_proj_head = use_proj_head
        self.proj_dim = proj_dim

        if use_proj_head:
            self.l1 = nn.Linear(self.repres_dim, self.repres_dim // 2)
            self.l2 = nn.Linear(self.repres_dim // 2, self.proj_dim)
            self.classifier = nn.Linear(self.proj_dim, self.n_class)
        else:
            self.classifier = nn.Linear(self.repres_dim, self.n_class)

        # remove the classifier of the original model
        self.features.classifier = nn.Sequential()

    def forward(self, x, return_features=False):
        h = self.features(x)
        if self.use_proj_head:
            h = self.l1(h)
            h = F.relu(h)
            h = self.l2(h)
        out = self.classifier(h)

        if return_features:
            return out, h
        else:
            return out


class MOON(FedAvg):

    def __init__(self, args, appr_args, logger, party_list_rounds,
                party2nets, global_net,
                party2loaders, global_train_dl, test_dl):
        super(MOON, self).__init__(args, appr_args, logger, party_list_rounds,
                party2nets, global_net,
                party2loaders, global_train_dl, test_dl)

        # re-wrap the models
        for party_id in self.party2nets:
            self.party2nets[party_id] = ContrastiveModelWrapper(
                                self.party2nets[party_id],
                                self.appr_args.use_proj_head,
                                self.appr_args.proj_dim)
        self.global_net = ContrastiveModelWrapper(
                self.global_net, self.appr_args.use_proj_head, self.appr_args.proj_dim)

        # store all local models of the last round
        self.party2prevnets = copy.deepcopy(self.party2nets)
        for _, net in self.party2prevnets.items():
            net.eval()
            for param in net.parameters():
                param.requires_grad = False

    # function that processing the special arguments of current method
    @staticmethod
    def extra_parser(extra_args):
        parser = ArgumentParser()
        # feddecorr arguments
        parser.add_argument('--feddecorr', action='store_true',
                            help='whether to use FedDecorr')
        parser.add_argument('--feddecorr_coef', type=float, default=0.1,
                            help='coefficient of the FedDecorr loss')

        # MOON parameters
        parser.add_argument('--mu', type=float, default=5.0,
                            help='the mu parameter for moon')
        parser.add_argument('--proj_dim', type=int, default=256,
                            help='projection dimension of the projector')
        parser.add_argument('--temperature', type=float, default=0.5,
                            help='the temperature parameter for contrastive loss')
        parser.add_argument('--use_proj_head', action='store_true',
                            help='whether to use projection head')
        return parser.parse_args(extra_args)


    def _local_training(self, party_id):
        # the three networks used in the algorithm
        # ==============================================
        net = self.party2nets[party_id]
        net.train()
        net.cuda()

        self.global_net.cuda()
        self.global_net.eval()

        prev_local_net = self.party2prevnets[party_id]
        prev_local_net.eval()
        prev_local_net.cuda()
        # ==============================================


        train_dataloader = self.party2loaders[party_id]
        test_dataloader = self.test_dl

        self.logger.info('Training network %s' % str(party_id))
        self.logger.info('n_training: %d' % len(train_dataloader))
        self.logger.info('n_test: %d' % len(test_dataloader))

        train_acc, _ = compute_accuracy(net, train_dataloader)
        test_acc, _ = compute_accuracy(net, test_dataloader)

        self.logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
        self.logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                              lr=self.args.lr, momentum=self.args.rho, weight_decay=self.args.weight_decay)

        # initialize all necessary loss
        criterion = nn.CrossEntropyLoss()
        feddecorr = FedDecorrLoss()
        cos_sim = torch.nn.CosineSimilarity(dim=-1)

        for epoch in range(self.args.epochs):
            epoch_loss_collector = []
            epoch_loss1_collector = []
            epoch_loss2_collector = []
            for batch_idx, (x, target) in enumerate(train_dataloader):
                x, target = x.cuda(), target.cuda()
                target = target.long()

                optimizer.zero_grad()

                out, features = net(x, return_features=True)
                _, features_global = self.global_net(x, return_features=True)
                _, features_prev_local = prev_local_net(x, return_features=True)

                # classification loss
                loss_cls = criterion(out, target)

                # contrastive loss
                # similarity of positive pair (i.e., w/ global model)
                pos_similarity = cos_sim(features, features_global).view(-1, 1)
                # similarity of negative pair (i.e., w/ previous round local model)
                neg_similarity = cos_sim(features, features_prev_local).view(-1, 1)
                repres_sim = torch.cat([pos_similarity, neg_similarity], dim=-1)
                contrast_label = torch.zeros(repres_sim.size(0)).long().cuda()
                loss_con = criterion(repres_sim, contrast_label)

                loss = loss_cls + self.appr_args.mu*loss_con

                # decorrelation loss (if applicable)
                if self.appr_args.feddecorr:
                    loss_feddecorr = feddecorr(features)
                    loss = loss + self.appr_args.feddecorr_coef * loss_feddecorr

                loss.backward()
                optimizer.step()

                epoch_loss_collector.append(loss.item())
                epoch_loss1_collector.append(loss_cls.item())
                epoch_loss2_collector.append(loss_con.item())

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
            epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
            self.logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % 
                             (epoch, epoch_loss, epoch_loss1, epoch_loss2))

        train_acc, _ = compute_accuracy(net, train_dataloader)
        test_acc, _ = compute_accuracy(net, test_dataloader)
        self.logger.info('>> Training accuracy: %f' % train_acc)
        self.logger.info('>> Test accuracy: %f' % test_acc)

        net.to('cpu')
        self.global_net.to('cpu')
        # store the current model as the "previous round local model"
        # to prepare for the next round
        self.party2prevnets[party_id] = copy.deepcopy(net)
