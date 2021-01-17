from torch import nn

from util.lt_util import cal_gain


class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, activation=None,
                 batch_norm=False, dropout=0, dropout_before=True, initial="normal"):
        super(MLPLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout_before = dropout_before
        self.initial = initial
        self.reset_parameters()

    def reset_parameters(self):
        if self.initial == "kaiming":
            self.linear.reset_parameters()
        else:
            gain = cal_gain(self.activation)
            if self.initial == "uniform":
                nn.init.xavier_uniform_(self.linear.weight, gain=gain)
            elif self.initial == "normal":
                nn.init.xavier_normal_(self.linear.weight, gain=gain)
            else:
                raise Exception("There is no initial: {}".format(self.initial))
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)
        if self.batch_norm:
            self.bn.reset_parameters()

    def forward(self, features):

        if self.dropout_before:
            h = self.dropout(features)
            h = self.linear(h)
            if self.batch_norm:
                h = self.bn(h)
            if self.activation:
                h = self.activation(h)
        else:
            h = self.linear(features)
            if self.batch_norm:
                h = self.bn(h)
            if self.activation:
                h = self.activation(h)
            h = self.dropout(h)

        return h