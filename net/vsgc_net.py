from layer.vsgc_layer import VSGCLayer
import torch.nn as nn


class VSGCNet(nn.Module):
    def __init__(self, num_feats, num_classes, bias=True, k=1, dropout=0, propagation=0):
        super(VSGCNet, self).__init__()
        self.vsgc = VSGCLayer(num_feats, num_classes, bias=bias, k=k, dropout=dropout, propagation=propagation)

    def forward(self, graph, features):
        h = self.vsgc(graph, features)
        return h