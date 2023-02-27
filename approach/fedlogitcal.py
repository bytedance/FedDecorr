# *************************************************************************
# Copyright 2023 ByteDance and/or its affiliates
#
# Copyright 2023 FedDecorr Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# *************************************************************************


import numpy as np
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from .fedavg import FedAvg

import sys
sys.path.insert(0, '../')
from utils import compute_accuracy

class FedLogitCal(FedAvg):

    def __init__(self, args, appr_args, logger, party_list_rounds,
                party2nets, global_net,
                party2loaders, global_train_dl, test_dl):
        super(FedLogitCal, self).__init__(args, appr_args, logger, party_list_rounds,
                party2nets, global_net,
                party2loaders, global_train_dl, test_dl)


    # function that processing the special arguments of current method
    @staticmethod
    def extra_parser(extra_args):
        parser = ArgumentParser()
        # logit calibration arguments
        parser.add_argument('--calibration_temp', type=float, default=0.1, help='calibration temperature')
        return parser.parse_args(extra_args)


    def _local_training(self, party_id):
        net = self.party2nets[party_id]
        net.train()
        net.cuda()

        train_dataloader = self.party2loaders[party_id]
        test_dataloader = self.test_dl

        # collect number of data per class on current local client
        n_class = net.classifier.out_features
        class2data = torch.zeros(n_class)
        ds = train_dataloader.dataset
        uniq_val, uniq_count = np.unique(ds.target, return_counts=True)
        for i, c in enumerate(uniq_val.tolist()):
            class2data[c] = uniq_count[i]
        class2data = class2data.unsqueeze(dim=0).cuda()

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

        for epoch in range(self.args.epochs):
            epoch_loss_collector = []
            for batch_idx, (x, target) in enumerate(train_dataloader):
                x, target = x.cuda(), target.cuda()
                target = target.long()
                optimizer.zero_grad()
                out = net(x)

                # calibrate logit
                out -= self.appr_args.calibration_temp * class2data**(-0.25)

                loss = criterion(out, target)
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

