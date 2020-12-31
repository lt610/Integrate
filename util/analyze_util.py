import pymongo
import matplotlib.pyplot as plt

def print_and_plot_vary_prop():
    client = pymongo.MongoClient("10.192.9.196:27017")
    db = client["sacred"]
    runs = db["runs"]
    query = {"experiment.name": "vsgc_vary_prop_cite"}
    items = runs.find(query).sort("config.params.k")

    ks = []
    accs = []
    part_idxs = [2, 4, 8, 16, 24, 32]
    part_accs = []
    for item in items:
        k = item["config"]["params"]["k"]
        ks.append(k)

        acc = eval(item["result"].split(" ± ")[0])
        accs.append(acc)
        if k in part_idxs:
            part_accs.append(acc)
    print(" ".join(list(map(str, part_accs))))
    plt.plot(ks, accs)
    plt.show()


if __name__ == "__main__":
    print_and_plot_vary_prop()