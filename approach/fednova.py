# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..  
# *************************************************************************

# Code ported from https://github.com/Xtra-Computing/NIID-Bench/blob/main/experiments.py


import copy
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

class FedNova(FedAvg):

    def __init__(self, args, appr_args, logger, party_list_rounds,
                party2nets, global_net,
                party2loaders, global_train_dl, test_dl):
        super(FedNova, self).__init__(args, appr_args, logger, party_list_rounds,
                party2nets, global_net,
                party2loaders, global_train_dl, test_dl)


    # function that processing the special arguments of current method
    @staticmethod
    def extra_parser(extra_args):
        parser = ArgumentParser()
        # feddecorr arguments
        parser.add_argument('--feddecorr', action='store_true',
                            help='whether to use FedDecorr')
        parser.add_argument('--feddecorr_coef', type=float, default=0.1,
                            help='coefficient of the FedDecorr loss')
        return parser.parse_args(extra_args)


    # function that executing the federated training
    def run_fed(self):
        for comm_round in range(self.args.n_comm_round):
            self.logger.info("in comm round:" + str(comm_round))

            # do local training on each party
            nets_this_round, all_a = self.local_training(comm_round)

            # conduct global aggregation
            self.global_aggregation(nets_this_round, all_a)

            # compute acc
            self.global_net.cuda()
            train_acc, train_loss = compute_accuracy(self.global_net, self.global_train_dl)
            test_acc, test_loss = compute_accuracy(self.global_net, self.test_dl)
            self.global_net.to('cpu')

            # logging numbers
            self.logger.info('>> Global Model Train accuracy: %f' % train_acc)
            self.logger.info('>> Global Model Test accuracy: %f' % test_acc)
            self.logger.info('>> Global Model Train loss: %f' % train_loss)

            if (comm_round + 1) % self.args.print_interval == 0:
                print('round: ', str(comm_round))
                print('>> Global Model Train accuracy: %f' % train_acc)
                print('>> Global Model Test accuracy: %f' % test_acc)
                print('>> Global Model Train loss: %f' % train_loss)

            if (comm_round+1) % self.args.save_interval == 0:
                torch.save(self.global_net.state_dict(),
                    os.path.join(self.args.ckptdir, self.args.approach, 'globalmodel_'+self.args.log_file_name+'.pth'))
                torch.save(self.party2nets[0].state_dict(),
                    os.path.join(self.args.ckptdir, self.args.approach, 'localmodel0_'+self.args.log_file_name+'.pth'))


    def global_aggregation(self, nets_this_round, all_a):
        total_data_points = sum([len(self.party2loaders[r].dataset) for r in nets_this_round])
        fed_avg_freqs = [len(self.party2loaders[r].dataset) / total_data_points for r in nets_this_round]

        global_w_init = copy.deepcopy(self.global_net.state_dict())
        global_w = self.global_net.state_dict()
        for net_id, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            if net_id == 0:
                for key in net_para:
                    global_w[key] = net_para[key] * fed_avg_freqs[net_id]
            else:
                for key in net_para:
                    global_w[key] += net_para[key] * fed_avg_freqs[net_id]

        # re-normalize global update
        tau_eff = sum([fed_avg_freqs[r]*all_a[r] for r in nets_this_round])
        for key in global_w:
            global_w[key] = global_w_init[key] + tau_eff*(global_w[key] - global_w_init[key])

        self.global_net.load_state_dict(global_w)


    def local_training(self, comm_round):
        # conduct local training on all selected clients
        party_list_this_round = self.party_list_rounds[comm_round]
        nets_this_round = {k: self.party2nets[k] for k in party_list_this_round}

        # send global model to all selected clients
        global_w = self.global_net.state_dict()
        for net in nets_this_round.values():
            net.load_state_dict(global_w)

        # enumerate all clients and train locally
        all_a = {party_id:0 for party_id in nets_this_round}
        for party_id in nets_this_round:
            all_a[party_id] = self._local_training(party_id)

        return nets_this_round, all_a


    def _local_training(self, party_id):
        net = self.party2nets[party_id]
        global_w = copy.deepcopy(net.state_dict())
        net.train()
        net.cuda()

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

        n_step = 0
        for epoch in range(self.args.epochs):
            epoch_loss_collector = []
            for batch_idx, (x, target) in enumerate(train_dataloader):
                x, target = x.cuda(), target.cuda()
                target = target.long()
                optimizer.zero_grad()
                out, features = net(x, return_features=True)

                loss = criterion(out, target)
                if self.appr_args.feddecorr:
                    loss_feddecorr = feddecorr(features)
                    loss = loss + self.appr_args.feddecorr_coef * loss_feddecorr
                loss.backward()
                optimizer.step()

                n_step += 1

                epoch_loss_collector.append(loss.item())

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            self.logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        train_acc, _ = compute_accuracy(net, train_dataloader)
        test_acc, _ = compute_accuracy(net, test_dataloader)
        self.logger.info('>> Training accuracy: %f' % train_acc)
        self.logger.info('>> Test accuracy: %f' % test_acc)
        net.to('cpu')

        # post processing by normalizing local update
        a_i = (n_step - self.args.rho * (1 - pow(self.args.rho, n_step)) / (1 - self.args.rho)) / (1 - self.args.rho)
        net_para = net.state_dict()
        for key in net_para:
            net_para[key] = global_w[key] + torch.true_divide(net_para[key] - global_w[key], a_i)
        net.load_state_dict(net_para)
        return a_i
