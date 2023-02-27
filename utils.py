# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..  
# *************************************************************************

# Code ported from https://github.com/QinbinLi/MOON/blob/main/utils.py


import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import random


# fixing every seed
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def model_parameter_vector(model):
    param = [p.view(-1) for p in model.parameters()]
    return torch.concat(param, dim=0)

def compute_accuracy(model, dataloader):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss()
    loss_collector = []
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = model(x)

            loss = criterion(out, target)
            _, pred_label = torch.max(out.data, 1)
            loss_collector.append(loss.item())
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

        avg_loss = sum(loss_collector) / len(loss_collector)

    if was_training:
        model.train()

    return correct / float(total), avg_loss


