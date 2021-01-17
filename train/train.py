import torch.nn.functional as F
from torch import nn

def train(model, graph, features, labels, train_mask, optimizer):
    model.train()
    logits = model(graph, features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[train_mask], labels[train_mask])
    # loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return model



