# MIT License
# Copyright (c) 2020 Vijay Prakash Dwivedi, Chaitanya K. Joshi, Thomas Laurent, Yoshua Bengio, Xavier Bresson


"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import random as rd
import torch.nn as nn
import math

from .metrics import accuracy_MNIST_CIFAR as accuracy

def train_epoch(model, optimizer, device, data_loader, epoch, augmentation):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_snorm_e = batch_snorm_e.to(device)
        batch_labels = batch_labels.to(device)
        batch_snorm_n = batch_snorm_n.to(device) # num x 1
        if augmentation:
            batch_graphs_aug = batch_graphs
            angle = (torch.random(batch_graphs_aug['eig'][:, 1].shape) - 0.5) / 4
            batch_graphs_aug['eig'][:, 1], batch_graphs_aug['eig'][:, 2] = (1 - angle)**(0.5) * batch_graphs_aug['eig'][:, 1] \
                                                                           + angle * batch_graphs_aug['eig'][:, 2], \
                                                                           (1 - angle)**(0.5) * batch_graphs_aug['eig'][:, 2] -  \
                                                                           angle * batch_graphs_aug['eig'][:, 1]
            print(batch_graphs_aug['eig'][0, 1])
            print(batch_graphs['eig'][0, 1])

        optimizer.zero_grad()
        if augmentation:
            batch_scores = model.forward(batch_graphs_aug, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
        else:
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    
    return epoch_loss, epoch_train_acc, optimizer

def evaluate_network(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_snorm_e = batch_snorm_e.to(device)
            batch_labels = batch_labels.to(device)
            batch_snorm_n = batch_snorm_n.to(device)
            
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
            loss = model.loss(batch_scores, batch_labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        
    return epoch_test_loss, epoch_test_acc