from layer.vsgc_layer import VSGCLayer
from layer.mlp_layer import MLPLayer
from net.mlp_net import MLPNet
import torch.nn as nn
import torch.nn.functional as F


class VSGCNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, bias=True, k=1, alp=1, lam=1, activation=F.relu,
                 batch_norm=False, dropout=0, dropout_before=True, propagation=0,
                 with_mlp=False, mlp_before=False):

        super(VSGCNet, self).__init__()
        self.with_mlp = with_mlp
        self.mlp_before = mlp_before
        if with_mlp:
            if mlp_before:
                map_in_dim, map_out_dim, map_acti = in_dim, hidden_dim, activation
            else:
                map_in_dim, map_out_dim, map_acti = in_dim, hidden_dim, None
            mlps_in_dim, mlps_hidden_dim, mlps_out_dim = hidden_dim, hidden_dim, out_dim
            self.mlps = MLPNet(
                layer_num=2,
                in_dim=mlps_in_dim,
                hidden_dim=mlps_hidden_dim,
                out_dim=mlps_out_dim,
                bias=bias,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
                dropout_before=dropout_before,
            )
        else:
            map_in_dim, map_out_dim, map_acti = in_dim, out_dim, None

        self.map = MLPLayer(
            in_dim=map_in_dim,
            out_dim=map_out_dim,
            batch_norm=batch_norm,
            dropout=dropout,
            dropout_before=dropout_before
        )

        self.vsgc = VSGCLayer(
            k=k,
            alp=alp,
            lam=lam,
            propagation=propagation
        )

    def forward(self, graph, features):

        if self.with_mlp:
            if self.mlp_before:
                h = self.map(features)
                h = self.mlps(h)
                h = self.vsgc(graph, h)
            else:
                h = self.map(features)
                h = self.vsgc(graph, h)
                h = self.mlps(h)
        else:
            h = self.map(features)
            h = self.vsgc(graph, h)
        return h


# class VSGCNet(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim, bias=True, k=1, activation=F.relu,
#                  batch_norm=False, dropout=0, dropout_before=True, propagation=0,
#                  with_mlp=False, mlp_before=False):
#
#         super(VSGCNet, self).__init__()
#
#         self.in_map = MLPLayer(
#             in_dim=in_dim,
#             out_dim=hidden_dim,
#             batch_norm=batch_norm,
#             dropout=dropout,
#             dropout_before=dropout_before
#         )
#
#         self.vsgc = VSGCLayer(
#             k=k,
#             propagation=propagation
#         )
#
#         self.out_map = MLPLayer(
#             in_dim=hidden_dim,
#             out_dim=out_dim,
#             batch_norm=batch_norm,
#             dropout=dropout,
#             dropout_before=dropout_before
#         )
#
#     def forward(self, graph, features):
#
#         h = self.in_map(features)
#         h = self.vsgc(graph, h)
#         h = self.out_map(h)
#         return h
