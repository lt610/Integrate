from torch import nn
from util.other_util import cal_gain


class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False, activation=None,
                 batch_norm=False, dropout=0):
        super(MLPLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = cal_gain(self.activation)
        nn.init.xavier_normal_(self.linear.weight, gain=gain)

    def forward(self, g, features):
        h_pre = features
        h = self.dropout(features)
        h = self.linear(h)
        if self.batch_norm:
            h = self.bn(h)
        if self.activation:
            h = self.activation(h)
        return h