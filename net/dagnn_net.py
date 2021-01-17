import torch.nn as nn
import torch.nn.functional as F
from layer.dagnn_layer import DAGNNLayer
from net.mlp_net import MLPNet


class DAGNNNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, k, bias=True, activation=F.relu,
                 batch_norm=False, dropout=0, dropout_before=True, initial="uniform"):
        super(DAGNNNet, self).__init__()
        self.mlp = MLPNet(2, in_dim, hidden_dim, out_dim, bias,
                          activation, batch_norm, dropout,
                          dropout_before, initial)
        self.dagnn = DAGNNLayer(out_dim, k)

    def forward(self, graph, features):
        h = self.mlp(features)
        h = self.dagnn(graph, h)
        return h