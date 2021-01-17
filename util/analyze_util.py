import pymongo
import matplotlib.pyplot as plt


def search_mongo(query, sort):
    client = pymongo.MongoClient("10.192.9.196:27017")
    db = client["sacred"]
    runs = db["runs"]
    items = items = runs.find(query).sort(sort)
    return items


def parse_acc(ex_name):
    query = {"experiment.name": ex_name}
    sort = "config.params.k"
    items = search_mongo(query, sort)
    all_idxs = []
    all_accs = []
    part_idxs = [2, 4, 8, 16, 24, 32]
    part_accs = []
    for item in items:
        k = item["config"]["params"]["k"]
        all_idxs.append(k)

        acc = eval(item["result"].split(" ± ")[0])
        all_accs.append(acc)
        if k in part_idxs:
            part_accs.append(item["result"])
    return all_idxs, all_accs, part_idxs, part_accs


def print_and_plot_vary_prop():

    title = "vary_prop_cora"

    ex_name = "vsgc_{}".format(title)
    all_idxs, all_accs, part_idxs, part_accs = parse_acc(ex_name)
    print("{}: {}".format(ex_name, ",".join(part_accs)))
    vsgc_accs = all_accs
    if title == "vary_prop_cora":
        exact_accs = [83.52 for _ in range(2, 33)]
    else:
        exact_accs = [73.15 for _ in range(2, 33)]

    ex_name = "sgc_{}".format(title)
    all_idxs, all_accs, part_idxs, part_accs = parse_acc(ex_name)
    print("{}: {}".format(ex_name, ",".join(part_accs)))
    sgc_accs = all_accs

    plt.title(title)
    plt.xlabel("prop step")
    plt.ylabel("test acc")
    plt.xlim(1.5, 32.5)

    marker = ''
    ms = 4
    plt.plot(all_idxs, vsgc_accs, color='green', linestyle='-', marker=marker, ms=ms, label="base")
    plt.plot(all_idxs, sgc_accs, color='orange', linestyle='-', marker=marker, ms=ms, label="sgc")
    plt.plot(all_idxs, exact_accs, color='blueviolet',linestyle='--', marker=marker, ms=ms, label="exact")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print_and_plot_vary_prop()