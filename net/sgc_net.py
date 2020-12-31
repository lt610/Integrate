from dgl.nn.pytorch.conv import SGConv
import torch.nn as nn
import torch.nn.functional as F
from layer.dagnn_layer import DAGNNLayer


class SGCNet(nn.Module):
    def __init__(self, in_dim, out_dim, k):
        super(SGCNet, self).__init__()
        self.sgc = SGConv(in_feats=in_dim, out_feats=out_dim, k=k, cached=True)

    def forward(self, graph, features):
        h = self.sgc(graph, features)
        return h
