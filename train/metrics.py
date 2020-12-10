import torch as th
import torch.nn.functional as F


def evaluate_acc(model, graph, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def evaluate_loss(model, graph, features, labels, mask):
    model.eval()
    with th.no_grad():
        """交叉熵"""
        logits = model(graph, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask])
        return loss.item()


def evaluate_single_acc_loss(model, graph, features, labels, mask):
    """合并后只要一次前向计算"""
    model.eval()
    with th.no_grad():
        logits = model(graph, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask])
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return loss.item(), correct.item() * 1.0 / len(labels)


def evaluate_all_acc_loss(model, graph, features, labels, masks):
    """在一次前向传播中计算多组值"""
    model.eval()
    with th.no_grad():
        logits = model(graph, features)
        logp = F.log_softmax(logits, 1)
        results = ()
        for mask in masks:
            loss = F.nll_loss(logp[mask], labels[mask])
            tmp_logits = logits[mask]
            tmp_labels = labels[mask]
            _, indices = th.max(tmp_logits, dim=1)
            correct = th.sum(indices == tmp_labels)
            results += (loss.item(), correct.item() * 1.0 / len(tmp_labels))
        return results
