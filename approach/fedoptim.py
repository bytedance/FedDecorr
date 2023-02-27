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


# This file includes implementation of both FedAvgM and FedAdam
# Essentially, FedAvgM is using vanilla gradient descent as server optimizer
# FedAdam is using Adam as server optimizer

import os
import copy
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from .fedavg import FedAvg

import sys
sys.path.insert(0, '../')
from utils import compute_accuracy

class FedOptim(FedAvg):

    def __init__(self, args, appr_args, logger, party_list_rounds,
                party2nets, global_net,
                party2loaders, global_train_dl, test_dl):
        super(FedOptim, self).__init__(args, appr_args, logger, party_list_rounds,
                party2nets, global_net,
                party2loaders, global_train_dl, test_dl)


    # function that processing the special arguments of current method
    @staticmethod
    def extra_parser(extra_args):
        parser = ArgumentParser()
        # FedAvgM, FedAdam, FedAdagrad
        parser.add_argument('--server_optimizer', type=str, default='gd',
                            help='the server optimizer. \
                            gd corresponds to FedAvgM and adam corresponds to FedAdam')
        parser.add_argument('--server_momentum', type=float, default=0.9,
                            help='the first order server momentum')
        parser.add_argument('--server_momentum_second', type=float, default=0.99,
                            help='the second order server momentum')
        parser.add_argument('--server_learning_rate', type=float, default=1.0,
                            help='Server learning rate of fedadam/fedyogi')
        parser.add_argument('--tau', type=float, default=0.001,
                            help='tau introduced in FedAdam paper. \
                            Essentially, this hyper-parameter provides \
                            numeric protection for second-order momentum')

        # feddecorr arguments
        parser.add_argument('--feddecorr', action='store_true',
                            help='whether to use FedDecorr')
        parser.add_argument('--feddecorr_coef', type=float, default=0.1,
                            help='coefficient of the FedDecorr loss')
        return parser.parse_args(extra_args)


    # function that executing the federated training
    def run_fed(self):
        assert self.appr_args.server_optimizer in ['gd', 'adagrad', 'adam', 'yogi'], \
            'server optimizer not implemented'

        # initialize server momentum
        moment_first = copy.deepcopy(self.global_net.state_dict())
        moment_second = copy.deepcopy(self.global_net.state_dict())
        for key in moment_first:
            moment_first[key].zero_()
            moment_second[key].zero_()

        for comm_round in range(self.args.n_comm_round):
            self.logger.info("in comm round:" + str(comm_round))

            # record original global model parameters
            old_w = copy.deepcopy(self.global_net.state_dict())

            # do local training on each party
            nets_this_round = self.local_training(comm_round)

            # conduct global aggregation
            self.global_aggregation(nets_this_round)

            # conduct server update
            global_w = self.global_net.state_dict()
            delta_w = copy.deepcopy(global_w)
            for key in delta_w:
                delta_w[key] = global_w[key] - old_w[key]

                # update first order moment
                moment_first[key] = self.appr_args.server_momentum * moment_first[key] + \
                    (1-self.appr_args.server_momentum) * delta_w[key]

                # update second order moment
                if self.appr_args.server_optimizer == 'adagrad':
                    moment_second[key] = moment_second[key] + delta_w[key]**2
                elif self.appr_args.server_optimizer == 'adam':
                    moment_second[key] = self.appr_args.server_momentum_second * moment_second[key] + \
                        (1 - self.appr_args.server_momentum_second) * (delta_w[key]**2)
                elif self.appr_args.server_optimizer == 'yogi':
                    moment_second[key] = moment_second[key] - \
                        (1 - self.appr_args.server_momentum_second)*(delta_w[key]**2) * \
                        torch.sign(moment_second[key]-delta_w[key]**2)

                # update global model parameters
                if self.appr_args.server_optimizer == 'gd':
                    # server learning rate is fixed to 1.0 regardless
                    global_w[key] = old_w[key] + moment_first[key]
                else:
                    # else it's second order algorithms
                    global_w[key] = old_w[key] + \
                        self.appr_args.server_learning_rate * \
                        moment_first[key]/(self.appr_args.tau+torch.sqrt(moment_second[key]))

            self.global_net.load_state_dict(global_w)

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

