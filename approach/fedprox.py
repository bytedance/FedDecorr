# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..  
# *************************************************************************

# Code ported from https://github.com/QinbinLi/MOON/blob/main/main.py


import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from .fedavg import FedAvg

import sys
sys.path.insert(0, '../')
from utils import compute_accuracy
from loss import FedDecorrLoss


class FedProx(FedAvg):

    def __init__(self, args, appr_args, logger, party_list_rounds,
                party2nets, global_net,
                party2loaders, global_train_dl, test_dl):
        super(FedProx, self).__init__(args, appr_args, logger, party_list_rounds,
                party2nets, global_net,
                party2loaders, global_train_dl, test_dl)


    # function that processing the special arguments of current method
    @staticmethod
    def extra_parser(extra_args):
        parser = ArgumentParser()
        # fedprox arguments
        parser.add_argument('--mu', type=float, default=0.001,
                            help='regularization coefficient of FedProx')

        # feddecorr arguments
        parser.add_argument('--feddecorr', action='store_true',
                            help='whether to use FedDecorr')
        parser.add_argument('--feddecorr_coef', type=float, default=0.1,
                            help='coefficient of the FedDecorr loss')
        return parser.parse_args(extra_args)

    def _local_training(self, party_id):
        net = self.party2nets[party_id]
        net.train()
        net.cuda()

        global_weight_collector = list(net.parameters())

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
        criterion = nn.CrossEntropyLoss()
        feddecorr = FedDecorrLoss()

        for epoch in range(self.args.epochs):
            epoch_loss_collector = []
            for batch_idx, (x, target) in enumerate(train_dataloader):
                x, target = x.cuda(), target.cuda()
                target = target.long()
                optimizer.zero_grad()
                out, features = net(x, return_features=True)

                # ce loss
                loss = criterion(out, target)

                # feddecorr loss
                if self.appr_args.feddecorr:
                    loss_feddecorr = feddecorr(features)
                    loss = loss + self.appr_args.feddecorr_coef * loss_feddecorr

                # fedprox loss                
                fed_prox_reg = 0.0
                for param_index, param in enumerate(net.parameters()):
                    fed_prox_reg += ((self.appr_args.mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss += fed_prox_reg

                loss.backward()
                optimizer.step()

                epoch_loss_collector.append(loss.item())

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            self.logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        train_acc, _ = compute_accuracy(net, train_dataloader)
        test_acc, _ = compute_accuracy(net, test_dataloader)
        self.logger.info('>> Training accuracy: %f' % train_acc)
        self.logger.info('>> Test accuracy: %f' % test_acc)
        net.to('cpu')
