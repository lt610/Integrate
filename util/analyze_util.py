import pymongo
import matplotlib.pyplot as plt


def search_mongo(query, sort):
    client = pymongo.MongoClient("10.192.9.196:27017")
    db = client["sacred"]
    runs = db["runs"]
    items = items = runs.find(query).sort(sort)
    return items


def parse_acc(ex_name, start, end):
    query = {"experiment.name": ex_name, "config.params.k": {"$in": [i for i in range(start, end+1)]}}
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
    start, end = 2, 100

    ex_name = "vsgc_{}__".format(title)
    all_idxs, all_accs, part_idxs, part_accs = parse_acc(ex_name, start, end)
    print("{}: {}".format(ex_name, ",".join(part_accs)))
    vsgc_accs = all_accs

    if title == "vary_prop_cora":
        exact_accs = [83.18 for _ in range(start, end + 1)]
    else:
        exact_accs = [73.14 for _ in range(start, end + 1)]

    ex_name = "sgc_{}".format(title)
    all_idxs, all_accs, part_idxs, part_accs = parse_acc(ex_name, start, end)
    print("{}: {}".format(ex_name, ",".join(part_accs)))
    sgc_accs = all_accs

    # plt.title(title)
    plt.figure(figsize=(9, 5))
    plt.rc('font', size=17)

    plt.xlabel("Propagation Steps")
    plt.ylabel("Test Accuracy")
    plt.xlim(start - 0.5, end + 0.5)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    marker = ''
    ms = 4

    plt.plot(all_idxs, vsgc_accs, color='green', linestyle='-', marker=marker, ms=ms, label="$TWIRLS_{base}$")
    plt.plot(all_idxs, sgc_accs, color='orange', linestyle='-', marker=marker, ms=ms, label="SGC")
    plt.plot(all_idxs, exact_accs, color='blueviolet',linestyle='--', marker=marker, ms=ms, label="Analytical")

    plt.legend()
    plt.savefig("../result/{}.eps".format(title))
    plt.show()


if __name__ == "__main__":
    print_and_plot_vary_prop()