import torch as th
from torch import nn
import dgl.function as fn


class VSGCLayer(nn.Module):
    def __init__(self, alp=1, lam=1, propagation=0):
        super(VSGCLayer, self).__init__()
        self.alp = alp
        self.lam = lam
        self.propagation = propagation

    def propagation_lt(self, graph, h, ini_h):
        g = graph.local_var()
        bef_A, aft_A, bef_X = g.ndata["bef_A"], g.ndata["aft_A"], g.ndata["bef_X"]
        ini_h = ini_h * bef_X
        pre_h = h

        g.ndata["h"] = h * aft_A
        g.update_all(fn.copy_u('h', 'm'),
                     fn.sum('m', 'h'))
        h = g.ndata.pop('h')
        h = h * bef_A

        h = self.alp * self.lam * h + (1 - self.alp) * pre_h + self.alp * ini_h
        return h

    def propagation_yyy(self, graph, h, ini_h):
        g = graph.local_var()
        D_in, D_out = g.ndata["D_in"], g.ndata["D_out"]
        pre_h = h

        g.ndata["h"] = h * D_out
        g.update_all(fn.copy_u('h', 'm'),
                     fn.sum('m', 'h'))
        h = g.ndata.pop('h')
        h = h * D_in

        h = self.alp * self.lam * h + (1 - self.alp * self.lam - self.alp) * pre_h + self.alp * ini_h
        return h

    def exact_solution(self, graph, h):
        h = graph.ndata["exact"].mm(h)
        return h

    def forward(self, graph, h, ini_h):
        if self.propagation == "lt":
            h = self.propagation_lt(graph, h, ini_h)
        elif self.propagation == "yyy":
            h = self.propagation_yyy(graph, h, ini_h)
        elif self.propagation == "exact":
            h = self.exact_solution(graph, h)
        return h