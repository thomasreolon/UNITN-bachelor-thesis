import mycommons.mqtt_wrap as mqtt
import mycommons.data_structure as ds
import pymongo
import numpy as np
import abc  # For abstract base classes.
import itertools  # For combinatorics
import resource
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
import scipy

# resource.setrlimit(resource.RLIMIT_DATA, (1024 ** 3, 1024 ** 3))

host = 'localhost'
port = 27017
pca_dim = 3
demos_rel = 25

args = {
    'sim_threshold': 0.43,
    'similarity': {
        'age':0.45,
        'gender':0.25,
        'other':0.30,
    }
}


def init(client):
    client.users = pymongo.MongoClient(host, port)['personas']['users']
    client.counter = {}

def get_attributes(user:ds.Insights):
    tags = []

    # ages tags
    if user.age<0:return None
    age = int(user.age)
    tags += [f'age-{x}' for x in [age-1, age, age+1]]
    tags.append(
        'age-'+str(age//10)+ ('0' if age%10<5 else '5') + 's'
    )

    # interests tags
    if user.interests:
        interests = sorted(user.interests.items(), key=lambda x: x[1], reverse=True)[:4]
        for interest, perc in interests: tags.append(interest)

    # gender tags
    if user.gender<0: return None
    tags.append('gender-'+('M' if user.gender>0.5 else 'F'))

    return tags

def process_request(client, cluster: ds.Cluster):
    client.counter[cluster.token] = 0
    client.model = DataClusterLinear(args['similarity'], args['sim_threshold'])

    # calc - cluster
    users = client.users.find({"token": cluster.token})
    for user in users:

        input = get_attributes(ds.Insights(user))
        if input:
            try:
                cluster_id = client.model.cluster(client.counter[cluster.token], input)
                if cluster_id == client.counter[cluster.token]:
                    client.counter[cluster.token] += 1
                    cluster.append_to_new_cluster(ds.Insights(user))
                else:
                    cluster.append(ds.Insights(user), cluster_id)
            except: pass
    if len(cluster.cluster) == 0:
        raise ds.UnableToProcessUser('token with 0 users associated')

    cluster.cluster.sort(key=lambda x:1/len(x))
    par = args["similarity"]
    cluster.set_type(f'mimosa(age={par["age"]}, gender={par["gender"]}, treshold={args["sim_threshold"]})')
    return cluster, f'tot={sum([len(x) for x in cluster.cluster])} cluster sizes = {[len(x) for x in cluster.cluster]}'


class DataCluster(abc.ABC):
    """Subclass must implement cluster() method."""

    @abc.abstractmethod
    def cluster(self, sig_num, elements):
        """The cluster() method must be implemented by a subclass."""
        pass


class DataClusterLinear(DataCluster, abc.ABC):
    """Implement a MIMOSA linear-time clustering algorithm."""

    def similarity_size(self, new, key):
        age, gender, other = set(), set(), set()
        age_n, gender_n, other_n = 0, 0, 0
        sim = self.similarity
        for k in new:
            if 'age' in k:
                age.add(k)
            elif 'gender' in k:
                gender.add(k)
            else:
                other.add(k)
        for k in key:
            if 'age' in k and k in age:
                age_n += 1
            elif 'gender' in k and k in gender:
                gender_n += 1
            elif k in other:
                other_n += 1

        return age_n/len(age)*sim['age'] + gender_n/len(gender)*sim['gender'] + other_n/(len(other)+1)*sim['other']

    def __init__(self, similarity, sim_treshold, markers={}):
        self.similarity = similarity
        self.sim_treshold = sim_treshold
        self.markers = markers

    def cluster(self, sig_num, elements):
        res, max_sim = sig_num, self.sim_treshold

        for cl_id in self.markers:
            sim = self.similarity_size(elements, self.markers[cl_id])
            if max_sim < sim:
                res, max_sim = cl_id, sim

        if sig_num == res:
            self.markers[sig_num] = elements

        return res



def main(name, id='0'):
    mqtt.MQTTClient(name, init=init, funct=process_request, id=id, type=mqtt.CLUSTER)

if __name__ == '__main__':
    # main('mimosa_clustering_pca')
    c = lambda: None
    init(c)
    x, res = process_request(c, ds.Cluster(token='theffballers'))
    print(res)

