from sacred import Experiment
from sacred.observers import MongoObserver
from train.metric import evaluate_all_acc_loss, evaluate_single_acc_loss
from train.prepare import prepare_data, prepare_model
from train.train import train, generate_random_seeds, \
    set_random_state, get_free_gpu, print_log_tvt, print_tvt, print_split
import torch as th

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

@ex.automain
def main(gpus, max_proc_num, seed, model_name, params):
    # 这里以后如果有需要可以封装到find_free_devices()中
    if not th.cuda.is_available():
        device = "cpu"
    else:
        device = get_free_gpu(gpus, max_proc_num)

    graph, features, labels, train_mask, val_mask, test_mask, num_feats, num_classes = prepare_data(device, params)
    random_seeds = generate_random_seeds(seed, params["num_runs"])
    test_accs = []

    for run in range(params["num_runs"]):
        set_random_state(random_seeds[run])
        model, optimizer, early_stopping = prepare_model(device, params, num_feats, num_classes, model_name)

        log_run_num = 2
        # 只记录前几个runs的logs
        if run < log_run_num:
            print_split(" {}th run ".format(run))

        for epoch in range(params['num_epochs']):
            train(model, graph, features, labels, train_mask, optimizer)
            train_loss, train_acc, val_loss, val_acc, test_loss, test_acc\
                = evaluate_all_acc_loss(model, graph, features, labels,
                                        (train_mask, val_mask, test_mask))

            if params["ex_by_loss"]:
                score = -val_loss
            else:
                score = val_acc

            early_stopping(score, (train_loss, train_acc, val_loss, val_acc,
                                     test_loss, test_acc))

            # current val_acc 和 test_acc
            cva, cta = early_stopping.metrics[3], early_stopping.metrics[5]
            # 只记录第一个run的epoch-loss等曲线，其他更个性化的图片结果可以记录在artifacts里面
            if run == 0:
                print_log_tvt(ex, epoch, train_loss, train_acc, val_loss, val_acc,
                              test_loss, test_acc, cva, cta)
            # 只记录前3 runs的logs, 这里有空封装一下吧
            elif run < log_run_num:
                print_tvt(epoch, train_loss, train_acc, val_loss, val_acc,
                          test_loss, test_acc, cva, cta)

            if early_stopping.is_stop:
                # 只记录前3 runs的logs
                if run < log_run_num:
                    print("Early stopping at epoch:{:04d}".format(epoch - 100))
                break
        train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = early_stopping.metrics
        # 只记录前3 runs的logs
        if run < log_run_num:

            print_tvt(epoch - 100, train_loss, train_acc, val_loss, val_acc,
                      test_loss, test_acc, cva, cta)
        test_accs.append(test_acc)

    print_split("\nFinal Result")
    avg = sum(test_accs) / params["num_runs"]
    std = ((sum([(t - avg) ** 2 for t in test_accs])) / params["num_runs"]) ** 0.5

    avg = round(avg * 100, 2)
    std = round(std * 100, 2)

    ex.log_scalar("avg_test_acc", avg)
    ex.log_scalar("std_test_acc", std)

    result = "{} ± {}".format(avg, std)
    print("Test_acc: {}".format(result))
    return result


