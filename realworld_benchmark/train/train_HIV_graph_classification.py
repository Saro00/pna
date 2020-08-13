"""
    Utility functions for training one epoch
    and evaluating one epoch
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl
import numpy as np
from ogb.graphproppred import Evaluator
from tqdm import tqdm

"""
    For GCNs
"""


def train_epoch_sparse(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_ROC = 0

    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        batch_scores = model.forward(batch_graphs, batch_x, batch_e, True, True)
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        evaluator = Evaluator(name='ogbg-molhiv')
        epoch_train_ROC += evaluator.eval({'y_pred': torch.sigmoid(batch_scores), 'y_true': batch_labels.type(torch.FloatTensor).to('cuda').unsqueeze(-1)})['rocauc']
    epoch_loss /= (iter + 1)
    epoch_train_ROC /= (iter + 1)

    return epoch_loss, epoch_train_ROC, optimizer


def evaluate_network_sparse(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_ROC = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, True, True)
            loss = model.loss(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()
            evaluator = Evaluator(name='ogbg-molhiv')
            epoch_test_ROC += evaluator.eval({'y_pred': torch.sigmoid(batch_scores), 'y_true': batch_labels.unsqueeze(-1)})['rocauc']
        epoch_test_loss /= (iter + 1)
        epoch_test_ROC /= (iter + 1)

    return epoch_test_loss, epoch_test_ROC
