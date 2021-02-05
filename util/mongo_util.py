import pymongo


def delete_records(host, dataset, collection, query):
    client = pymongo.MongoClient(host)
    db = client[dataset]
    col = db[collection]
    res = col.delete_many(query)
    print("{}条记录已删除".format(res.deleted_count))


host = "127.0.0.1:27017"
dataset = "sacred"
collection = "runs"
query = {"experiment.name": {"$eq": "vsgc_nonlinear_geom_cita"}}
delete_records(host, dataset, collection, query)

