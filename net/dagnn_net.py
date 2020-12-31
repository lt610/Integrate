import torch.nn as nn
import torch.nn.functional as F
from util.train_util import cal_gain
from layer.dagnn_layer import DAGNNLayer


class DAGNNNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, k, bias=True, activation=F.relu, dropout=0):
        super(DAGNNNet, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias)
        self.linear2 = nn.Linear(hidden_dim, out_dim, bias)
        self.dagnn = DAGNNLayer(out_dim, k)
        self.activation = activation
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        gain = cal_gain(self.activation)
        nn.init.xavier_uniform_(self.linear1.weight, gain=gain)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear2.bias)
        nn.init.xavier_uniform_(self.linear1.weight)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, graph, features):
        h = F.dropout(features, self.dropout, training=self.training)
        h = self.activation(self.linear1(h))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.linear2(h)
        h = self.dagnn(graph, h)
        return h