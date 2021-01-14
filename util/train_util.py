from torch import nn
from torch.nn import functional as F
import torch as th
import dgl.function as fn


def cal_gain(fun, param=None):
    gain = 1
    if fun is F.sigmoid:
        gain = nn.init.calculate_gain('sigmoid')
    if fun is F.tanh:
        gain = nn.init.calculate_gain('tanh')
    if fun is F.relu:
        gain = nn.init.calculate_gain('relu')
    if fun is F.leaky_relu:
        gain = nn.init.calculate_gain('leaky_relu', param)
    return gain


def compute_D_and_e(g, lam, propagation):
    if propagation == "lt":
        degs = g.in_degrees().float() - 1.0
        norm_lambd_1 = 1 / (lam * degs + 1).unsqueeze(1)
        norm05 = (degs + 1).sqrt().unsqueeze(1)
        norm_05 = 1 / norm05

        g.ndata["bef_A"] = norm_lambd_1 * norm05
        g.ndata['aft_A'] = norm_05
        g.ndata['bef_X'] = norm_lambd_1
        # 这里是后续扩展到大图训练的时候用到
        # g.apply_edges(fn.u_mul_v('bef_A', 'aft_A', 'e'))
        # g.update_all(fn.copy_e('e', 't'), fn.sum('t', 'D_tilde'))
        # g.edata['e'] = g.edata['e'].view(g.num_edges(), 1)
    elif propagation == "yyy":
        g.ndata['D_in'] = 1 / g.in_degrees().float().sqrt().unsqueeze(1)
        g.ndata['D_out'] = 1 / g.out_degrees().float().sqrt().unsqueeze(1)
        # 这里是后续扩展到大图训练的时候用到
        # g.apply_edges(fn.u_mul_v('D_in', 'D_out', 'e'))
        # g.update_all(fn.copy_e('e', 't'), fn.sum('t', 'D_tilde'))
        # g.edata['e'] = g.edata['e'].view(g.num_edges(), 1)
    elif propagation == "exact":
        adj = g.adjacency_matrix().to(g.device)
        adj = (- th.eye(adj.shape[0]).to(g.device)).add(adj)
        degs = g.in_degrees().float().to(g.device)
        bef_A = degs.sqrt().diag()
        aft_A = (1 / degs.sqrt()).diag()
        sub_1 = (lam * degs + 1 - lam).diag()
        sub_2 = lam * bef_A.mm(adj.mm(aft_A))
        sub = sub_1 - sub_2
        g.ndata["exact"] = sub.inverse()
