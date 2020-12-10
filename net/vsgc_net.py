from layer.vsgc_layer import VSGCLayer
from layer.mlp_layer import MLPLayer
from net.mlp_net import MLPNet
import torch.nn as nn
import torch.nn.functional as F

class VSGCNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, bias=True, k=1, activation=F.relu,
                 batch_norm=False, dropout=0, propagation=0, mlp_layer_num=1):
        super(VSGCNet, self).__init__()
        self.mlps = MLPNet(
            layer_num=mlp_layer_num,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            bias=bias,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        self.vsgc = VSGCLayer(
            k=k,
            propagation=propagation
        )

    def forward(self, graph, features):
        h = self.mlps(features)
        h = self.vsgc(graph, h)
        return h
