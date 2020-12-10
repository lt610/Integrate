import torch.nn as nn
from layer.mlp_layer import MLPLayer
import torch.nn.functional as F


class MLPNet(nn.Module):
    def __init__(self, layer_num, in_dim, hidden_dim, out_dim, bias=True, activation=F.relu,
                 batch_norm=False, dropout=0):
        super(MLPNet, self).__init__()
        self.mlps = nn.ModuleList()
        for i in range(layer_num):
            if layer_num == 1:
                in_d, out_d, acti = in_dim, out_dim, None
            elif i == 0:
                in_d, out_d, acti = in_dim, hidden_dim, activation
            elif i == layer_num - 1:
                in_d, out_d, acti = hidden_dim, out_dim, None
            else:
                in_d, out_d, acti = hidden_dim, hidden_dim, activation
            self.mlps.append(MLPLayer(in_dim=in_d,
                                      out_dim=out_d,
                                      bias=bias,
                                      activation=acti,
                                      batch_norm=batch_norm,
                                      dropout=dropout))

    def forward(self, features):
        h = features
        for mlp in self.mlps:
            h = mlp(h)
        return h
