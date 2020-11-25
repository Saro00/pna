# MIT License
# Copyright (c) 2020 Vijay Prakash Dwivedi, Chaitanya K. Joshi, Thomas Laurent, Yoshua Bengio, Xavier Bresson


import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np


def MAE(scores, targets):
    #MAE = F.l1_loss(scores, torch.cuda.FloatTensor([[x] for x in targets]))
    #MAE = nn.MSELoss()(scores, targets)
    MAE = F.l1_loss(scores, targets)
    return MAE
