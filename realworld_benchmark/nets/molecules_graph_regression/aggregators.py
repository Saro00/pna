import torch
from functools import partial

EPS = 1e-5


def aggregate_mean(h, eig_s, eig_d):
  return torch.mean(h, dim=1)


def aggregate_max(h, eig_s, eig_d):
    return torch.max(h, dim=1)[0]


def aggregate_min(h, eig_s, eig_d):
    return torch.min(h, dim=1)[0]


def aggregate_std(h, eig_s, eig_d):
    return torch.sqrt(aggregate_var(h, eig_s, eig_d) + EPS)


def aggregate_var(h, eig_s, eig_d):
    h_mean_squares = torch.mean(h * h, dim=-2)
    h_mean = torch.mean(h, dim=-2)
    var = torch.relu(h_mean_squares - h_mean * h_mean)
    return var


def aggregate_moment(h, eig_s, eig_d, n=3):
    # for each node (E[(X-E[X])^n])^{1/n}
    # EPS is added to the absolute value of expectation before taking the nth root for stability
    h_mean = torch.mean(h, dim=1, keepdim=True)
    h_n = torch.mean(torch.pow(h - h_mean, n))
    rooted_h_n = torch.sign(h_n) * torch.pow(torch.abs(h_n) + EPS, 1. / n)
    return rooted_h_n


def aggregate_moment_3(h, eig_s, eig_d):
    return aggregate_moment(h, n=3)


def aggregate_moment_4(h, eig_s, eig_d):
    return aggregate_moment(h, n=4)


def aggregate_moment_5(h, eig_s, eig_d):
    return aggregate_moment(h, n=5)


def aggregate_sum(h, eig_s, eig_d):
    return torch.sum(h, dim=1)

def aggregate_eig(h, eig_s, eig_d, eig_idx):
    #check right unsqueeze...
    h_mod = torch.mul(h, (torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx])/(torch.sum(torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]), dim=1, keepdim=True) + EPS)).unsqueeze(-1))
    return torch.sum(h_mod, dim=1)

def aggregate_eig_bis(h, eig_s, eig_d, eig_idx):
    #check right unsqueeze...
    h_mod = torch.mul(h, (eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]/(torch.sum(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx], dim=1, keepdim=True) + EPS)).unsqueeze(-1))
    return torch.abs(torch.sum(h_mod, dim=1))

def aggregate_NN(h, eig_filt):
    h_mod = torch.mul(h, eig_filt)
    return torch.sum(h_mod, dim=1)



AGGREGATORS = {'mean': aggregate_mean, 'sum': aggregate_sum, 'max': aggregate_max, 'min': aggregate_min,
               'std': aggregate_std, 'var': aggregate_var, 'moment3': aggregate_moment_3, 'moment4': aggregate_moment_4,
               'moment5': aggregate_moment_5,  'eig1-smooth': partial(aggregate_eig, eig_idx=1),
               'eig2-smooth': partial(aggregate_eig, eig_idx=2), 'eig3-smooth': partial(aggregate_eig, eig_idx=3),
               'eig1-dx': partial(aggregate_eig_bis, eig_idx=1), 'eig2-dx': partial(aggregate_eig_bis, eig_idx=2),
               'eig3-dx': partial(aggregate_eig_bis, eig_idx=3), 'aggregate_NN': aggregate_NN}