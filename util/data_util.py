from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset


def load_citation_data(dataset_name):
    if dataset_name == 'cora':
        dataset = CoraGraphDataset()
    elif dataset_name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif dataset_name == 'pubmed':
        dataset = PubmedGraphDataset()
    else:
        raise Exception("The dataset {} doesn't exist.".format(dataset_name))
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
