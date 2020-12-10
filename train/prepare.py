from net.asgc_net import ASGCNet
from net.vsgc_net import VSGCNet
from train.early_stopping import EarlyStopping
from util.data_util import load_citation_data
import torch as th


def prepare_data(device, params):
    graph, features, labels, train_mask, \
    val_mask, test_mask, num_feats, num_classes = load_citation_data(params['dataset'])
    labels = labels.squeeze()
    graph = graph.to(device)
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    return graph, features, labels, train_mask, \
           val_mask, test_mask, num_feats, num_classes


def prepare_model(device, params, num_feats, num_classes, model_name):
    if model_name == 'asgc':
        model = ASGCNet(
            num_feats=num_feats,
            num_classes=num_classes,
            num_hidden=params['num_hidden'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        )
    elif model_name == "vsgc":
        model = VSGCNet(
            num_feats=num_feats,
            num_classes=num_classes,
            k=params["k"],
            dropout=params["dropout"],
        )
    else:
        pass
    model = model.to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    early_stopping = EarlyStopping(params['patience'])
    return model, optimizer, early_stopping
