from net.sgc_net import SGCNet
from net.vsgc_net import VSGCNet
from net.dagnn_net import DAGNNNet
from train.early_stopping import EarlyStopping
from util.data_util import load_data
import torch as th
from util.train_util import compute_D_and_e


def prepare_data(device, params):
    graph, features, labels, train_mask, \
    val_mask, test_mask, num_feats, num_classes = load_data(params['dataset'])
    labels = labels.squeeze()

    graph = graph.to(device)
    if "propagation" in params:
        compute_D_and_e(graph, params['lam'], params["propagation"])

    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    return graph, features, labels, train_mask, \
           val_mask, test_mask, num_feats, num_classes


def prepare_model(device, params, num_feats, num_classes, model_name):
    if model_name == "vsgc":
        model = VSGCNet(
            in_dim=num_feats,
            hidden_dim=params["hidden_dim"],
            out_dim=num_classes,
            k=params["k"],
            alp=params["alp"],
            lam=params["lam"],
            batch_norm=params["batch_norm"],
            drop_map=params["drop_map"],
            drop_mlp=params["drop_mlp"],
            dropout_before=params["dropout_before"],
            propagation=params["propagation"],
            with_mlp=params["with_mlp"],
            mlp_before=params["mlp_before"]
        )
    elif model_name == "dagnn":
        model = DAGNNNet(
            in_dim=num_feats,
            hidden_dim=params["hidden_dim"],
            out_dim=num_classes,
            k = params["k"],
            batch_norm=params["batch_norm"],
            dropout=params["dropout"],
            dropout_before=params["dropout_before"]
        )
    elif model_name == "sgc":
        model = SGCNet(
            in_dim=num_feats,
            out_dim=num_classes,
            k=params["k"]
        )
    else:
        pass
    model = model.to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    early_stopping = EarlyStopping(params['patience'])
    return model, optimizer, early_stopping


