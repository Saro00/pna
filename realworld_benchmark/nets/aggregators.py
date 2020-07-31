import torch
from functools import partial


EPS = 1e-5


def aggregate_mean(self, h, eig_s, eig_d):
  return torch.mean(h, dim=1)


def aggregate_max(self, h, eig_s, eig_d):
    return torch.max(h, dim=1)[0]


def aggregate_min(self, h, eig_s, eig_d):
    return torch.min(h, dim=1)[0]


def aggregate_std(self, h, eig_s, eig_d):
    return torch.sqrt(aggregate_var(h, eig_s, eig_d) + EPS)


def aggregate_var(self, h, eig_s, eig_d):
    h_mean_squares = torch.mean(h * h, dim=-2)
    h_mean = torch.mean(h, dim=-2)
    var = torch.relu(h_mean_squares - h_mean * h_mean)
    return var


def aggregate_moment(self, h, eig_s, eig_d, n=3):
    # for each node (E[(X-E[X])^n])^{1/n}
    # EPS is added to the absolute value of expectation before taking the nth root for stability
    h_mean = torch.mean(h, dim=1, keepdim=True)
    h_n = torch.mean(torch.pow(h - h_mean, n))
    rooted_h_n = torch.sign(h_n) * torch.pow(torch.abs(h_n) + EPS, 1. / n)
    return rooted_h_n


def aggregate_moment_3(self, h, eig_s, eig_d):
    return aggregate_moment(h, n=3)


def aggregate_moment_4(self, h, eig_s, eig_d):
    return aggregate_moment(h, n=4)


def aggregate_moment_5(self, h, eig_s, eig_d):
    return aggregate_moment(h, n=5)


def aggregate_sum(self, h, eig_s, eig_d):
    return torch.sum(h, dim=1)

def aggregate_eig(self, h, eig_s, eig_d, eig_idx):
    h_mod = torch.mul(h, (torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx])/
                      (torch.sum(torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]), keepdim=True, dim=1) + EPS)).unsqueeze(-1))
    return torch.sum(h_mod, dim=1)

def aggregate_eig_softmax(self, h, eig_s, eig_d, eig_idx, alpha):
    h_mod = torch.mul(h, torch.nn.Softmax(1)(alpha * (torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx])).unsqueeze(-1)))
    return torch.sum(h_mod, dim=1)

def aggregate_eig_dir(self, h, eig_s, eig_d, eig_idx):
    h_mod = torch.mul(h, (torch.abs(eig_s[:, :, eig_idx]) - torch.abs(eig_d[:, :, eig_idx]))/
                      (torch.sum(torch.abs(eig_s[:, :, eig_idx]) - torch.abs(eig_d[:, :, eig_idx]), keepdim=True, dim=1))).unsqueeze(-1)
    return torch.sum(h_mod, dim=1)

def aggregate_eig_bis2(self, h, eig_s, eig_d, eig_idx):
    h_mod = torch.mul(h, torch.relu(-torch.mul(torch.sign(eig_s[:, :, eig_idx]).unsqueeze(-1),
            ((eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx])/(torch.sum(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx], keepdim=True, dim=1) + EPS)).unsqueeze(-1))))
    return torch.sum(h_mod, dim=1)

def aggregate_NN(h, eig_filt):
    h_mod = torch.mul(h, eig_filt)
    return torch.sum(h_mod, dim=1)



AGGREGATORS = {'mean': aggregate_mean, 'sum': aggregate_sum, 'max': aggregate_max, 'min': aggregate_min,
               'std': aggregate_std, 'var': aggregate_var, 'moment3': aggregate_moment_3, 'moment4': aggregate_moment_4,
               'moment5': aggregate_moment_5,  'eig1-smooth': partial(aggregate_eig, eig_idx=1),
               'eig2-smooth': partial(aggregate_eig, eig_idx=2), 'eig3-smooth': partial(aggregate_eig, eig_idx=3),
                'eig4-smooth': partial(aggregate_eig, eig_idx=4),'eig5-smooth': partial(aggregate_eig, eig_idx=5),
                'eig1-dir' : partial(aggregate_eig_dir, eig_idx=1), 'eig2-dir' : partial(aggregate_eig_dir, eig_idx=2),
                'eig3-dir' : partial(aggregate_eig_dir, eig_idx=3), 'eig1-1' : partial(aggregate_eig_softmax, eig_idx=1, alpha=1),
                'eig1-0.5' : partial(aggregate_eig_softmax, eig_idx=1, alpha=0.5), 'eig1-0.1' : partial(aggregate_eig_softmax, eig_idx=1, alpha=0.1),
                'eig1-2' : partial(aggregate_eig_softmax, eig_idx=1, alpha=2), 'eig1-5' : partial(aggregate_eig_softmax, eig_idx=1, alpha=5),
                'eig1-10' : partial(aggregate_eig_softmax, eig_idx=1, alpha=10), 'eig2-1' : partial(aggregate_eig_softmax, eig_idx=2, alpha=1),
                'eig2-0.5' : partial(aggregate_eig_softmax, eig_idx=2, alpha=0.5), 'eig2-0.1' : partial(aggregate_eig_softmax, eig_idx=2, alpha=0.1),
                'eig2-2' : partial(aggregate_eig_softmax, eig_idx=2, alpha=2), 'eig2-5' : partial(aggregate_eig_softmax, eig_idx=2, alpha=5),
                'eig2-10' : partial(aggregate_eig_softmax, eig_idx=2, alpha=10),
                'eig1-50' : partial(aggregate_eig_softmax, eig_idx=1, alpha=50), 'eig2-50' : partial(aggregate_eig_softmax, eig_idx=2, alpha=50),
                'eig1-100' : partial(aggregate_eig_softmax, eig_idx=1, alpha=100), 'eig2-100' : partial(aggregate_eig_softmax, eig_idx=2, alpha=100),
                'eig1-0.01' : partial(aggregate_eig_softmax, eig_idx=1, alpha=0.01), 'eig2-0.01' : partial(aggregate_eig_softmax, eig_idx=2, alpha=0.01),
                'eig1-0.001' : partial(aggregate_eig_softmax, eig_idx=1, alpha=0.001), 'eig2-0.001' : partial(aggregate_eig_softmax, eig_idx=2, alpha=0.001),
                'eig1-neg-1' : partial(aggregate_eig_softmax, eig_idx=1, alpha=-1), 'eig2-neg-1' : partial(aggregate_eig_softmax, eig_idx=2, alpha=-1),
                'eig1-neg-10' : partial(aggregate_eig_softmax, eig_idx=1, alpha=-10), 'eig2-neg-10' : partial(aggregate_eig_softmax, eig_idx=2, alpha=-10),
                'eig1-neg-0.1' : partial(aggregate_eig_softmax, eig_idx=1, alpha=-0.1), 'eig2-neg-0.1' : partial(aggregate_eig_softmax, eig_idx=2, alpha=-0.1),
                'eig1-down' : partial(aggregate_eig_bis2, eig_idx=1), 'eig2-down' : partial(aggregate_eig_bis2, eig_idx=2),
                'aggregate_NN': aggregate_NN}