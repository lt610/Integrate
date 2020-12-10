from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset


# 后续还可以加入full—supervised等
def load_data(dataset_name):
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        return load_citation_data(dataset_name)
    elif dataset_name in ['ogbn-arxiv']:
        return load_ogb_data(dataset_name)
    else:
        raise Exception("The dataset {} doesn't exist.".format(dataset_name))


def load_citation_data(dataset_name):
    if dataset_name == 'cora':
        dataset = CoraGraphDataset()
    elif dataset_name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif dataset_name == 'pubmed':
        dataset = PubmedGraphDataset()

    graph = dataset[0]
    graph = graph.remove_self_loop().add_self_loop()
    print(graph)
    features = graph.ndata['feat']
    labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    num_feats = features.shape[1]
    num_classes = int(labels.max().item() + 1)
    return graph, features, labels, train_mask, val_mask, test_mask, num_feats, num_classes


def load_ogb_data(dataset_name):
    dataset = DglNodePropPredDataset(name=dataset_name)
    splitted_mask = dataset.get_idx_split()
    train_mask, val_mask, test_mask = splitted_mask['train'], splitted_mask['valid'], splitted_mask['test']
    graph, labels = dataset[0]
    features = graph.ndata["feat"]
    num_feats = features.shape[1]
    num_classes = (labels.max() + 1).item()
    # add reverse edges
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)
    # add self-loop
    graph = graph.remove_self_loop().add_self_loop()

    return graph, features, labels, train_mask, val_mask, test_mask, num_feats, num_classes