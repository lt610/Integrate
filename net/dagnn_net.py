import torch.nn as nn
import torch.nn.functional as F
from layer.dagnn_layer import DAGNNLayer
from net.mlp_net import MLPNet


class DAGNNNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, k, bias=True, activation=F.relu,
                 batch_norm=False, dropout=0, dropout_before=True):
        super(DAGNNNet, self).__init__()
        self.mlp = MLPNet(layer_num=2, in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                          bias=bias, activation=activation, batch_norm=batch_norm, dropout=dropout,
                          dropout_before=dropout_before)
        self.dagnn = DAGNNLayer(out_dim=out_dim, k=k)

    def forward(self, graph, features):
        h = self.mlp(features)
        h = self.dagnn(graph, h)
        return h