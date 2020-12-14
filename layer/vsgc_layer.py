import torch as th
from torch import nn
import dgl.function as fn


class VSGCLayer(nn.Module):
    def __init__(self, k=1, alp=1, lam=1, propagation=0):
        super(VSGCLayer, self).__init__()
        self.alp = alp
        self.lam = lam
        self.k = k
        self.propagation = propagation

    def forward(self, graph, features):
        g = graph.local_var()
        degs = g.in_degrees().float() - 1.0
        norm_lambd_1 = th.pow(self.lam * degs + 1.0, -1)
        norm_lambd_1 = norm_lambd_1.to(features.device).unsqueeze(1)

        norm05 = th.pow(degs + 1.0, 0.5)
        norm05 = norm05.to(features.device).unsqueeze(1)
        norm_05 = th.pow(degs + 1.0, -0.5)
        norm_05 = norm_05.to(features.device).unsqueeze(1)

        h = features
        h_pre = h
        h_initial = h * norm_lambd_1
        if self.propagation == 0:
            for _ in range(self.k):
                h = h * norm_05

                g.ndata['h'] = h
                g.update_all(fn.copy_u('h', 'm'),
                             fn.sum('m', 'h'))
                h = g.ndata.pop('h')

                h = h * norm_lambd_1 * norm05

                h = self.alp * self.lam * h + (1 - self.alp) * h_pre + self.alp * h_initial

                h_pre = h

        else:
            pass

        return h

    # def forward(self, graph, features):
    #     g = graph.local_var()
    #     degs = g.in_degrees().float()
    #
    #     norm_1 = th.pow(degs, -1).to(features.device).unsqueeze(1)
    #     norm_05 = th.pow(degs, -0.5).to(features.device).unsqueeze(1)
    #
    #     h = features
    #     if self.propagation == 0:
    #         h_initial = h * norm_1
    #         for _ in range(self.k):
    #             h = h * norm_05
    #             g.ndata['h'] = h
    #             g.update_all(fn.copy_u('h', 'm'),
    #                          fn.sum('m', 'h'))
    #             h = g.ndata.pop('h')
    #             h = h * norm_05
    #             h = h + h_initial
    #
    #     elif self.propagation == 1:
    #         h_pre = h
    #         h_initial = h
    #         for _ in range(self.k):
    #             h = h * norm_05
    #             g.ndata['h'] = h
    #             g.update_all(fn.copy_u('h', 'm'),
    #                          fn.sum('m', 'h'))
    #             h = g.ndata.pop('h')
    #             h = h * norm_05
    #             h = h + h_initial - h_pre
    #             h_pre = h
    #     elif self.propagation == 2:
    #         h_pre = h
    #         h_initial = h * norm_1
    #         for _ in range(self.k):
    #             h = h * norm_05
    #             g.ndata['h'] = h
    #             g.update_all(fn.copy_u('h', 'm'),
    #                          fn.sum('m', 'h'))
    #             h = g.ndata.pop('h')
    #             h = h * norm_05
    #             h = h + h_initial - h_pre
    #             h_pre = h
    #     elif self.propagation == 3:
    #         h_initial = h
    #         for _ in range(self.k):
    #             h = h * norm_05
    #             g.ndata['h'] = h
    #             g.update_all(fn.copy_u('h', 'm'),
    #                          fn.sum('m', 'h'))
    #             h = g.ndata.pop('h')
    #             h = h * norm_05
    #             h = h + h_initial
    #
    #     return h