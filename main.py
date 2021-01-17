from sacred import Experiment
from sacred.observers import MongoObserver
from train.metric import evaluate_all_acc_loss, evaluate_single_acc_loss
from train.prepare import prepare_data, prepare_model
from train.train import train
from util.lt_util import get_free_gpu, generate_random_seeds, set_random_state, \
    log_split, log_metric, log_rec_metric, rec_metric

ex = Experiment()
    ex.observers.append(MongoObserver(url='10.192.9.196:27017',
                                      db_name='sacred'))

@ex.config
def base_config():
    tags = "debug"
    config_name = "dagnn"
    if tags == "debug":
        ex.add_config('config/base_config/{}.json'.format(config_name))
    elif tags == "final":
        ex.add_config("config/best_config/{}.json".format(config_name))
    elif tags == "search":
        ex.add_config("config/search_config/{}.json".format(config_name))
    elif tags == "analyze":
        ex.add_config("config/analyze_config/{}.json".format(config_name))
    else:
        raise Exception("There is no {}".format(tags))
    ex_name = config_name
    model_name = config_name.split("_")[0]


def compute_final_result(ex, params, test_accs):
    log_split("Final Result")
    avg = sum(test_accs) / params["num_runs"]
    std = ((sum([(t - avg) ** 2 for t in test_accs])) / params["num_runs"]) ** 0.5

    avg = round(avg * 100, 2)
    std = round(std * 100, 2)

    ex.log_scalar("avg_test_acc", avg)
    ex.log_scalar("std_test_acc", std)

    result = "{:.2f} ± {:.2f}".format(avg, std)
    print("Test_acc: {}".format(result))
    return result


@ex.automain
def main(gpus, max_proc_num, seed, model_name, params):

    device = get_free_gpu(gpus, max_proc_num)
    random_seeds = generate_random_seeds(seed, params["num_runs"])
    test_accs = []

    for run in range(params["num_runs"]):
        set_random_state(random_seeds[run])
        graph, features, labels, train_mask, val_mask, test_mask, num_feats, num_classes = prepare_data(device, params, run)
        model, optimizer, early_stopping = prepare_model(device, params, num_feats, num_classes, model_name)

        if run == 0:
            log_split(" {}th run ".format(run))

        for epoch in range(params['num_epochs']):
            train(model, graph, features, labels, train_mask, optimizer)
            train_loss, train_acc, val_loss, val_acc, test_loss, test_acc\
                = evaluate_all_acc_loss(model, graph, features, labels,
                                        (train_mask, val_mask, test_mask))

            score = -val_loss if params["ex_by_loss"] else val_acc

            metric = {"Train Loss": train_loss, "Val Loss": val_loss, "Test Loss": test_loss,
                      "Train Acc": train_acc, "Val Acc": val_acc, "Test Acc": test_acc}

            early_stopping(score, metric)
            metric["Cur Acc"] = early_stopping.metrics["Test Acc"]
            if run == 0:
                log_rec_metric(ex, epoch, 4, metric)

            if early_stopping.is_stop:
                if run == 0:
                    print("Early stopping at epoch:{:04d}".format(epoch - params["patience"]))
                break

        metric = early_stopping.metrics

        print("Best Results of run {}".format(run))
        epoch = epoch if epoch < params["patience"] else epoch - params["patience"]
        log_metric(epoch, 4, **metric)
        rec_metric(ex, run, 4, **{"Run Acc": metric["Test Acc"]})

        test_accs.append(metric["Test Acc"])

    return compute_final_result(ex, params, test_accs)



