from torch import nn

from util.train_util import cal_gain


class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, activation=None,
                 batch_norm=False, dropout=0, dropout_before=True):
        super(MLPLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout_before = dropout_before
        self.reset_parameters()

    def reset_parameters(self):
        gain = cal_gain(self.activation)
        # nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

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