from layer.vsgc_layer import VSGCLayer
from layer.mlp_layer import MLPLayer
from net.mlp_net import MLPNet
import torch.nn as nn
import torch.nn.functional as F


class VSGCNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, bias=True, k=1, alp=1, lam=1, activation=F.relu,
                 batch_norm=False, drop_map=0, drop_mlp=0, dropout_before=True, propagation=0,
                 with_mlp=False, mlp_before=False):

        super(VSGCNet, self).__init__()
        self.propagation = propagation
        self.with_mlp = with_mlp
        self.mlp_before = mlp_before
        if with_mlp:
            if mlp_before:
                mlps_in_dim, mlps_hidden_dim, mlps_out_dim = in_dim, hidden_dim, out_dim
            else:
                map_in_dim, map_out_dim = in_dim, hidden_dim
                mlps_in_dim, mlps_hidden_dim, mlps_out_dim = hidden_dim, hidden_dim, out_dim
            self.mlps = MLPNet(layer_num=2, in_dim=mlps_in_dim, hidden_dim=mlps_hidden_dim,
                               out_dim=mlps_out_dim, bias=bias, activation=activation,
                               batch_norm=batch_norm, dropout=drop_mlp, dropout_before=dropout_before,
                               initial="normal", gain=False)
        else:
            map_in_dim, map_out_dim = in_dim, out_dim
        if not mlp_before:
            self.map = MLPLayer(in_dim=map_in_dim, out_dim=map_out_dim, batch_norm=batch_norm,
                                dropout=drop_map, dropout_before=dropout_before,
                                initial="normal", gain=False)
        if self.propagation == "exact":
            self.vsgcs = VSGCLayer(alp=alp, lam=lam, propagation=propagation)
        else:
            self.vsgcs = nn.ModuleList([VSGCLayer(alp=alp, lam=lam, propagation=propagation) for _ in range(k)])

    def iter_vsgc(self, graph, h, ini_h):
        for vsgc in self.vsgcs:
            h = vsgc(graph, h, ini_h)
        return h

    def forw_vsgc(self, graph, h, ini_h):
        if self.propagation == "exact":
            h = self.vsgcs(graph, h, ini_h)
        else:
            h = self.iter_vsgc(graph, h, ini_h)
        return h

    def forward(self, graph, features):
        if self.with_mlp:
            if self.mlp_before:
                h = self.mlps(features)
                h = self.forw_vsgc(graph, h, h)
            else:
                h = self.map(features)
                h = self.forw_vsgc(graph, h, h)
                h = self.mlps(h)
        else:
            h = self.map(features)
            h = self.forw_vsgc(graph, h, h)
        return h

