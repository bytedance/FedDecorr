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
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import importlib
import logging
import os
import copy
import datetime
import random

from networks import resnet20, resnet32, resnet18, mobilenetv2
from utils import *
from dataset_utils import partition_data, get_dataloader


def get_args():
    # general arguments for all methods
    parser = argparse.ArgumentParser()

    # federated setup parameters
    parser.add_argument('--model', type=str, default='mobilenetv2',
                        help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset used for training')
    parser.add_argument('--partition', type=str, default='homo',
                        help='the data partitioning strategy')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='concentration parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--n_parties', type=int, default=10,
                        help='number of workers in a distributed cluster')
    parser.add_argument('--sample_fraction', type=float, default=1.0,
                        help='how many clients are sampled in each round')
    parser.add_argument('--approach', type=str, default='fedavg',
                        help='federated learning algorithm being used')
    parser.add_argument('--n_comm_round', type=int, default=50,
                        help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0,
                        help="Random seed")

    # local training parameters
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--rho', type=float, default=0.9,
                        help='momentum of sgd optimizer')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of local epochs')   
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help="weight decay during local training")
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='the optimizer')
    parser.add_argument('--auto_aug', action='store_true',
                        help='whether to apply auto augmentation')

    # logging parameters
    parser.add_argument('--print_interval', type=int, default=50,
                        help='how many comm round to print results on screen')
    parser.add_argument('--datadir', type=str, required=False, default="./data/",
                        help="Data directory")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/",
                        help='Log directory path')
    parser.add_argument('--log_file_name', type=str, default=None,
                        help='The log file name')

    parser.add_argument('--ckptdir', type=str, required=False, default="./models/",
                        help='directory to save model')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='how many rounds do we save the checkpoint one time')

    args, appr_args = parser.parse_known_args()
    return args, appr_args


def init_nets(n_parties, args):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset in {'mnist', 'cifar10', 'svhn'}:
        n_classes = 10
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    else:
        raise NotImplementedError('dataset not supported')

    for net_i in range(n_parties):
        if args.model == 'resnet20':
            net = resnet20(num_classes=n_classes)
        elif args.model == 'resnet32':
            net = resnet32(num_classes=n_classes)
        elif args.model == 'resnet18':
            net = resnet18(num_classes=n_classes)
        elif args.model == 'mobilenetv2':
            net = mobilenetv2(num_classes=n_classes)
        else:
            raise NotImplementedError('model not supported')
        nets[net_i] = net

    return nets


if __name__ == '__main__':
    # ===== parsing arguments and initialize method =====
    args, appr_args = get_args()

    if args.approach == 'fedavg':
        from approach.fedavg import FedAvg as Appr
    elif args.approach == 'fedprox':
        from approach.fedprox import FedProx as Appr
    elif args.approach == 'fedsam':
        from approach.fedsam import FedSAM as Appr
    elif args.approach == 'fedlogitcal':
        from approach.fedlogitcal import FedLogitCal as Appr
    elif args.approach == 'fedrs':
        from approach.fedrs import FedRS as Appr
    elif args.approach == 'fedoptim':
        from approach.fedoptim import FedOptim as Appr
    elif args.approach == 'fednova':
        from approach.fednova import FedNova as Appr
    elif args.approach == 'moon':
        from approach.moon import MOON as Appr
    elif args.approach == 'fedexp':
        from approach.fedexp import FedExp as Appr
    else:
        raise NotImplementedError('approach not implemented')

    # arguments specific to the chosen FL algorithm
    appr_args = Appr.extra_parser(appr_args)
    # ===================================================


    # ================ logging related ==================
    mkdirs(args.logdir)
    mkdirs(args.ckptdir)
    mkdirs(os.path.join(args.ckptdir, args.approach))

    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    argument_path = argument_path + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args) + str(appr_args), f)
    print(str(args))
    print(str(appr_args))

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    print('log path: ', log_path)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')

    logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    # ===================================================


    # ================ dataset related ==================
    logger.info("Partitioning data")
    seed_everything(args.init_seed)
    # mapping from individual client to sample idx of the whole dataset
    party2dataidx = partition_data(
        args.dataset, args.datadir, args.partition, args.n_parties, alpha=args.alpha)
    # mapping from individual client to its local training data loader
    party2loaders = {}
    for party_id in range(args.n_parties):
        train_dl_local, _ = get_dataloader(args, args.dataset, args.datadir,
            args.batch_size, args.batch_size, party2dataidx[party_id])
        party2loaders[party_id] = train_dl_local
    # these loaders are used for evaluating accuracy of global model
    global_train_dl, test_dl = get_dataloader(args, args.dataset, args.datadir,
                            train_bs=args.batch_size, test_bs=args.batch_size)

    # support random party sampling
    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.n_comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.n_comm_round):
            party_list_rounds.append(party_list)
    # ===================================================


    # ================ network related ================
    logger.info("Initializing nets")
    party2nets = init_nets(args.n_parties, args)
    global_net = init_nets(1, args)[0]
    # =================================================

    # ================ run FL ================
    fed_alg = Appr(args, appr_args, logger, party_list_rounds,
                party2nets, global_net,
                party2loaders, global_train_dl, test_dl)
    fed_alg.run_fed()
    # ========================================
