import torch
from torch import nn
from functools import partial


EPS = 1e-8


def aggregate_mean(self, h):
  return torch.mean(h, dim=1)


def aggregate_max(self, h):
    return torch.max(h, dim=1)[0]


def aggregate_min(self, h):
    return torch.min(h, dim=1)[0]


def aggregate_var(self, h):
    h_mean_squares = torch.mean(h * h, dim=-2)
    h_mean = torch.mean(h, dim=-2)
    var = torch.relu(h_mean_squares - h_mean * h_mean)
    return var


def aggregate_moment(self, h, n=3):
    # for each node (E[(X-E[X])^n])^{1/n}
    # EPS is added to the absolute value of expectation before taking the nth root for stability
    h_mean = torch.mean(h, dim=1, keepdim=True)
    h_n = torch.mean(torch.pow(h - h_mean, n))
    rooted_h_n = torch.sign(h_n) * torch.pow(torch.abs(h_n) + EPS, 1. / n)
    return rooted_h_n


def aggregate_moment_3(self, h):
    return aggregate_moment(h, n=3)


def aggregate_moment_4(self, h):
    return aggregate_moment(h, n=4)


def aggregate_moment_5(self, h):
    return aggregate_moment(h, n=5)


def aggregate_sum(self, h):
    return torch.sum(h, dim=1)


def aggregate_lap(self, h, h_in):
    deg = h.shape[1]
    return torch.sum(h, dim=1) - h_in * deg


def aggregate_mean_abs(self, h):
  return torch.abs(torch.mean(h, dim=1))



AGGREGATORS = {'mean': aggregate_mean, 'sum': aggregate_sum, 'max': aggregate_max, 'min': aggregate_min,
               'std': aggregate_std, 'var': aggregate_var, 'moment3': aggregate_moment_3, 'moment4': aggregate_moment_4,
               'moment5': aggregate_moment_5,  'lap': aggregate_lap, 'mean_abs': aggregate_mean_abs}