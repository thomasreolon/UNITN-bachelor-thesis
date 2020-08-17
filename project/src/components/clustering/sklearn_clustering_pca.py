import mycommons.mqtt_wrap as mqtt
import mycommons.data_structure as ds
import pymongo
import numpy as np
import abc  # For abstract base classes.
import itertools  # For combinatorics
import resource
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import scipy
import matplotlib.pyplot as plt

# resource.setrlimit(resource.RLIMIT_DATA, (1024 ** 3, 1024 ** 3))

host = 'localhost'
port = 27017

pca_dim = 2

threshold = 15

def init(client):
    client.users = pymongo.MongoClient(host, port)['personas']['users']
    client.counter = {}

def get_point(user, pca:PCA, scaler, stats):
    m, M = stats
    del user.mbti
    del user.language
    # del user.nlu
    if not user.interests or user.age<0 or 'AI' not in user.interests:
        return None

    point = user.get_point()
    point = scaler.transform([point])[0]
    if not all(np.isfinite(point)):      # drop if nan
        return None
    point = np.array(pca.transform([point])[0])

    point = (point-m)*100/(M-m) * pca.explained_variance_ratio_
    return point


def get_pca(cursor):
    pca = PCA(n_components=pca_dim, svd_solver='arpack') # gli altri solver non vanno!! fortran schifoh
    scaler = StandardScaler()
    clf_out = EllipticEnvelope(contamination=0.1)

    X = []
    to_average = 200
    for user in cursor:
        if not 'age' in user or not user['age'] or  user['age']<0 or 'AI' not in user['interests']:continue
        point = ds.Insights(user)
        del point.mbti
        del point.language
        if len(point.interests)>0:
            X.append(point.get_point())

        to_average -= 1
        if to_average <= 0: break
    del cursor

    scaler = scaler.fit(X)
    X = scaler.transform(X)

    try:
        pca = pca.fit(X)
    except Exception as e: print(e) ; exit(0)
    X = pca.transform(X)
    clf_out.fit(X)
    pred = clf_out.predict(X)
    X = np.array( [x for x, pr in zip(X, pred) if pr>0] )

    # pca how
    # plt.matshow(pca.components_, cmap='viridis')
    # plt.yticks([0, 1], ['1st Comp', '2nd Comp'], fontsize=10)
    # plt.colorbar()
    # features = [
    #     'age', 'is male', 'animals', 'music', 'health', 'war', 'clothing', 'sports', 'drinks', 'rich', 'cosplay',
    #     'office', 'travel', 'family', 'baby',
    #     'basketball', 'video games', 'food', 'american football', 'dance', 'basketball', 'tennis', 'religion', 'biking',
    #     'tech', 'academics', 'activism', 'animalsNLU', 'cars', 'animation', 'soccer', 'nature', 'AI', 'design',
    #     'politics'
    # ]
    # plt.xticks(range(len(features)), features, rotation=65, ha='left')
    # plt.tight_layout()
    # plt.show()

    return pca, scaler, ( np.min(X, axis=0), np.max(X, axis=0) )


def process_request(client, cluster: ds.Cluster):
    client.counter[cluster.token] = 0

    # preproess data - dimensional reduction
    users = client.users.find({"token": cluster.token})
    pca, stdscal, stats = get_pca(users)

    # calc - cluster
    users = client.users.find({"token": cluster.token})
    inputs = []
    data = []
    for user in users:
        inp = get_point(ds.Insights(user), pca, stdscal, stats)
        if inp is not None:
            inputs.append(inp)
            data.append(ds.Insights(user))

    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold).fit(np.array(inputs))
    max_lab = 0
    for lab in clustering.labels_:
        if lab > max_lab:
            max_lab = lab
    clusters = [[] for _ in range(0, max_lab+1)]

    for lab, insight in zip(clustering.labels_, data):
        clusters[lab].append(insight)
    cluster.cluster = clusters

    cluster.cluster.sort(key=lambda x:1/len(x))
    cluster.set_type(f'agglomerative_pca(threshold={threshold})')
    return cluster, f'tot={sum([len(x) for x in cluster.cluster])} cluster sizes = {[len(x) for x in cluster.cluster]}'


def main(name, id='0'):
    mqtt.MQTTClient(name, init=init, funct=process_request, id=id, type=mqtt.CLUSTER)

if __name__ == '__main__':
    # main('mimosa_clustering_pca')
    c = lambda: None
    init(c)
    x, res = process_request(c, ds.Cluster(token='theffballers'))
    print(res)

