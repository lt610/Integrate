from sacred import Experiment
from sacred.observers import MongoObserver
from train.metrics import evaluate_all_acc_loss, evaluate_single_acc_loss
from train.prepare import prepare_data, prepare_model
from train.train import train, print_tvt, print_log_tvt, print_split, generate_random_seeds,\
    set_random_state, get_free_gpu
import torch as th

ex = Experiment()
ex.observers.append(MongoObserver(url='10.192.9.196:27017',
                                      db_name='sacred'))
models = ["asgc", "vsgc"]

@ex.config
def base_config():
    tags = "debug"
    model_name = "vsgc"
    if model_name not in models:
        raise Exception("The model {} doesn't exist.".format(model_name))
    ex.add_config("config/base_config/{}.json".format(model_name))


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
        model, optimizer, early_stopping = prepare_model(device, params, num_feats, num_classes, model_name)

        # 只记录前3 runs的logs
        if run < 3:
            print_split(" {}th run ".format(run))

        set_random_state(random_seeds[run])
        for epoch in range(params['num_epochs']):
            train(model, graph, features, labels, train_mask, optimizer)
            train_loss, train_acc, val_loss, val_acc, test_loss, test_acc\
                = evaluate_all_acc_loss(model, graph, features, labels, (train_mask, val_mask, test_mask))

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
            elif run < 2:
                print_tvt(epoch, train_loss, train_acc, val_loss, val_acc,
                          test_loss, test_acc, cva, cta)

            if early_stopping.is_stop:
                # 只记录前3 runs的logs
                if run < 2:
                    print("Early stopping at epoch:{:05d}".format(epoch - 100))
                break
        train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = early_stopping.metrics
        # 只记录前3 runs的logs
        if run < 2:
            # 这里的bug有空修改下
            print_tvt(epoch - 100, train_loss, train_acc, val_loss, val_acc,
                      test_loss, test_acc, cva, cta)
        test_accs.append(test_acc)

    print_split("Final Result")
    avg = sum(test_accs) / params["num_runs"]
    std = ((sum([(t - avg) ** 2 for t in test_accs])) / params["num_runs"]) ** 0.5

    avg = round(avg * 100, 2)
    std = round(std * 100, 2)

    ex.log_scalar("avg_test_acc", avg)
    ex.log_scalar("std_test_acc", std)

    result = "{} ± {}".format(avg, std)
    print("Test_acc: {}".format(result))
    return result


