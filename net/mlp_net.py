import torch.nn as nn
from layer.mlp_layer import MLPLayer
import torch.nn.functional as F


class MLPNet(nn.Module):
    def __init__(self, layer_num, in_dim, hidden_dim, out_dim, bias=True, activation=F.relu,
                 batch_norm=False, dropout=0, dropout_before=True, initial="uniform", gain=True):
        super(MLPNet, self).__init__()
        if layer_num < 2:
            raise Exception("The layer_num must larger than 1.")
        self.mlps = nn.ModuleList()
        for i in range(layer_num):
            bn, do = batch_norm, dropout
            if i == 0:
                in_d, out_d, acti = in_dim, hidden_dim, activation
            elif i == layer_num - 1:
                in_d, out_d, acti = hidden_dim, out_dim, None
            else:
                in_d, out_d, acti = hidden_dim, hidden_dim, activation
            self.mlps.append(MLPLayer(in_dim=in_d,
                                      out_dim=out_d,
                                      bias=bias,
                                      activation=acti,
                                      batch_norm=bn,
                                      dropout=do,
                                      dropout_before=dropout_before,
                                      initial=initial,
                                      gain=gain))

    def forward(self, features):
        h = features
        for mlp in self.mlps:
            h = mlp(h)
        return h
