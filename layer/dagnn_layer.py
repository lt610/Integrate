import torch as th
from torch import nn
from torch.nn import Parameter
import dgl.function as fn
from torch.nn import functional as F
from util.other_util import cal_gain


class DAGNNLayer(nn.Module):
    def __init__(self, out_dim, k, graph_norm=True):
        super(DAGNNLayer, self).__init__()
        self.s = Parameter(th.FloatTensor(out_dim, 1))
        self.k = k
        self.graph_norm = graph_norm
        self.reset_parameters()

    def reset_parameters(self):
        gain = cal_gain(F.sigmoid)
        nn.init.xavier_uniform_(self.s, gain=gain)


    def forward(self, graph, features):
        g = graph.local_var()
        h = features
        results = [h]

        if self.graph_norm:
            degs = g.in_degrees().float().clamp(min=1)
            norm = th.pow(degs, -0.5)
            norm = norm.to(features.device).unsqueeze(1)

        for _ in range(self.k):
            if self.graph_norm:
                h = h * norm
            g.ndata['h'] = h
            g.update_all(fn.copy_u('h', 'm'),
                         fn.sum('m', 'h'))
            h = g.ndata.pop('h')
            if self.graph_norm:
                h = h * norm
            results.append(h)
        H = th.stack(results, dim=1)
        S = F.sigmoid(th.matmul(H, self.s))
        S = S.permute(0, 2, 1)
        H = th.matmul(S, H).squeeze()
        return H