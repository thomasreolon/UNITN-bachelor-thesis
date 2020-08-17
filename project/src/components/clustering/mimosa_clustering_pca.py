import mycommons.mqtt_wrap as mqtt
import mycommons.data_structure as ds
import pymongo
import numpy as np
import abc  # For abstract base classes.
import itertools  # For combinatorics
import resource
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy

# resource.setrlimit(resource.RLIMIT_DATA, (1024 ** 3, 1024 ** 3))

host = 'localhost'
port = 27017
pca_dim = 3
demos_rel = 25

args = {
    'sim_threshold': 0.35,
    'sizes': [pca_dim*(pca_dim+1)//2+3],
    'similarity': 'jaccard'
}


def init(client):
    client.users = pymongo.MongoClient(host, port)['personas']['users']
    client.counter = {}

def get_tag(i, p, m, M):
    tag=[]
    p = int((p-m)*100/M)


    for slice in range(10, 10*(pca_dim+1-i), 10):
        tag.append(
            str(i+1)+'PCA'+str((p-(slice//3))//slice)
        )
    return tag

def get_attributes(user, pca:PCA, scaler, mins, maxs):
    del user.mbti
    del user.language
    # del user.nlu
    if not user.interests or user.age<0 or 'AI' not in user.interests:
        return None
    age = int(user.age)
    gender = 'M' if user.gender>0.5 else 'F'
    tags = [gender, str((age+3)//6), str(age//6)]

    point = user.get_point()
    point = scaler.transform([point])[0]
    if not all(np.isfinite(point)):      # drop if nan
        return None

    point = pca.transform([point])[0]

    for i, p in enumerate(point):
        m = mins[i]
        M = maxs[i]-m

        tg = get_tag(i, p, m, M)
        if not tg:
            return None
        else:
            tags += tg
    return tags

def clean_array(X):
    new = []
    med = np.median(X,axis=0)
    mad = scipy.stats.median_absolute_deviation(X, axis=0)
    for x in X:
        if all(np.isfinite(x)) and abs(x[0]-med[0])<mad[0]*3/2:
            new.append(x)
    if len(new)==0:
        raise ValueError('not good, len=0')

    # add a fake input to assure std is not 0
    l = len(X[0])
    new.append([-1]*l)

    return np.array(new)



def get_pca(cursor):
    pca = PCA(n_components=pca_dim, svd_solver='arpack') # gli altri solver non vanno!! fortran schifoh
    scaler = StandardScaler()
    X = []
    to_average = 200
    for user in cursor:
        if not 'age' in user or not user['age'] or user['age']<0 or 'AI' not in user['interests']:continue
        point = ds.Insights(user)
        # del point.ocean
        # del point.needs
        # del point.nlu
        del point.mbti
        del point.language
        if len(point.interests)>0:
            X.append(point.get_point())
        to_average -= 1
        if to_average <= 0: break
    del cursor
    X = clean_array(X)
    scaler = scaler.fit(X)
    X = scaler.transform(X)

    try:
        pca = pca.fit(X)
        X = pca.transform(X)
    except Exception as e: print(e) ; exit(0)

    avg, std, Y = np.median(X,axis=0), X.std(axis=0), []
    for x in X:
        for par, av, st in zip(x, avg, std):
            if abs(par-av)<st:
                Y.append(x)
    Y = np.array(Y)

    return pca, scaler, np.min(Y, axis=0), np.max(Y, axis=0)


def process_request(client, cluster: ds.Cluster):
    client.counter[cluster.token] = 0
    client.model = DataClusterLinear(args)

    # preproess data - dimensional reduction
    users = client.users.find({"token": cluster.token})
    pca, stdscal, mins, maxs = get_pca(users)

    # calc - cluster
    users = client.users.find({"token": cluster.token})
    for user in users:
        input = get_attributes(ds.Insights(user), pca, stdscal, mins, maxs)
        # insight.link = 'https://twitter.com/' + user['username']
        if input is not None:
            cluster_id = client.model.cluster(client.counter[cluster.token], input)
            if cluster_id == client.counter[cluster.token]:
                client.counter[cluster.token] += 1
                cluster.append_to_new_cluster(ds.Insights(user))
            else:
                cluster.append(ds.Insights(user), cluster_id)
    if len(cluster.cluster) == 0:
        raise ds.UnableToProcessUser('token with 0 users associated')

    cluster.cluster.sort(key=lambda x:1/len(x))
    cluster.set_type(f'mimosa_pca(treshold={args["sim_threshold"]})')
    return cluster, f'tot={sum([len(x) for x in cluster.cluster])} cluster sizes = {[len(x) for x in cluster.cluster]}'


class DataCluster(abc.ABC):
    """Subclass must implement cluster() method."""

    @abc.abstractmethod
    def cluster(self, sig_num, elements):
        """The cluster() method must be implemented by a subclass."""
        pass


class DataClusterLinear(DataCluster, abc.ABC):
    """Implement a MIMOSA linear-time clustering algorithm."""

    def similarity_size(self, n_mark_in, n_match_out, n_overlap):
        if callable(self.similarity):
            return self.similarity(n_mark_in, n_match_out, n_overlap)
        if self.similarity == 'jaccard':
            return n_overlap / (n_mark_in + n_match_out - n_overlap)
        else:
            raise ValueError('a type must be provided for the similarity ')

    def __init__(self, args):
        """Populate data needed for MIMOSA clustering.
        Arguments:
        args -- A dictionary whose keys are to become attributes of
                the DataClusterLinear object.  Required keys are:
                sim_threshold, sizes."""
        if not args.get('sizes'): raise ValueError('Sizes must be provided')
        for key in args.keys(): setattr(self, key, args[key])
        self.markers = {}

        # Invoke helper function to construct the MatchOut and MarkIn tables.
        self.mi_table, self.mo_table = self.make_mimo_table(self.sim_threshold,
                                                            self.sizes)

        # Precompute all partial signature combinations for each possible
        #   overlap size around signatures of each allowable size.
        self.combinations = [[] for sig_size in range(self.sizes[-1] + 1)]
        for sig_size in self.sizes:
            for ov_size in range(self.mi_table[sig_size][-1] + 1):
                self.combinations[sig_size].append([])
        for sig_size in self.sizes:  # Allowable signature sizes.
            nums = list(range(sig_size))
            for ov_size in self.mi_table[sig_size]:  # Overlap sizes.
                if ov_size > sig_size: continue  # Skip overlaps that are too big.

                # Store index values for the combination elements into a table.
                self.combinations[sig_size][ov_size] = list(
                    itertools.combinations(nums, ov_size))  ############TOO MANY COMBINATIONS

    def make_mimo_table(self, sim_threshold, sizes):
        """Construct and initialize MatchOut and MarkIn tables.
        Arguments:
        sim_threshold -- Similarity threshold value between 0 and 1.
        sizes -- List of allowable sizes of input signatures."""
        mark_in_table = [[] for i in range(sizes[-1] + 1)]
        match_out_table = [[] for i in range(sizes[-1] + 1)]

        # Nested loops through the sizes of two signatures, and the size of their overlap.
        for mi_size in sizes:

            for mo_size in sizes:

                for ov_size in range(1, 1 + (mi_size if mi_size < mo_size else
                mo_size)):

                    # Skip if the similarity size score doesn't meet the threshold.
                    similarity = self.similarity_size(mi_size, mo_size, ov_size)
                    if similarity < sim_threshold: continue

                    # Add [size,overlap] pair to MO table.
                    match_out_table[mo_size].append([mi_size, ov_size])

                    # Add overlap size to MI table. (But don't add duplicates.)
                    if not (mark_in_table[mi_size] and
                            any([x == ov_size
                                 for x in mark_in_table[mi_size]])):
                        mark_in_table[mi_size].append(ov_size)

                    break  # Only add the smallest matching overlap; skip the larger ones.

        return mark_in_table, match_out_table  # Return the two constructed tables.

    def cluster(self, sig_num, elements):
        """Take an input data signature and assign it to a cluster.
        This is the core function of the MIMOSA implementation.
        Arguments:
        sig_num -- Index of the current input, in input series.
        sim_threshold -- Similarity threshold, between 0 and 1.
        sizes -- List of allowable sizes of signatures.
        elements -- The elements of the input signature."""

        sig_size = len(elements)  # The signature size.
        combos_sig = self.combinations[sig_size]  # Find the needed portions
        mo_sig_table = self.mo_table[sig_size]  # of precomputed tables
        mi_sig_table = self.mi_table[sig_size]  # for this sig size.
        markers = self.markers  # Storage for markers.
        cluster_id = sig_num  # Initialize the cluster assignment.

        # Build the partial sigs needed for this signature.
        partial_sigs = [[] for ov_size in range(mi_sig_table[-1] + 1)]
        for ov_size in mi_sig_table:

            # Map the precomputed index values for each combination ...
            for combo in combos_sig[ov_size]:
                # ... into the actual elements of this signature.
                partial_sigs[ov_size].append(
                    "-".join([elements[i] for i in combo]))

        # Match-Out stage.
        # For each possible size of another signature that could be similar to the signature ...
        for row in mo_sig_table:

            mi_size, ov_size = list(row)

            # ... check whether any of the partial signatures for that size is marked.
            for psig in partial_sigs[ov_size]:

                # Construct an MO key by concatenating a size value with the partial
                #   signature, and check whether the key is in the hash table.
                mo_token = str(mi_size) + "-" + psig
                marker_cluster_id = markers.get(mo_token)
                if not marker_cluster_id: continue

                # If more than one MO key is found, assign the one that is
                #   marked with the earliest-numbered cluster.
                if marker_cluster_id < cluster_id:
                    cluster_id = marker_cluster_id

        # Mark-In stage.
        if cluster_id == sig_num:  # If an existing cluster was not found ...

            # ... construct MI keys for each possible overlap size for another
            #   signature that could be similar to the signature.
            for ov_size in mi_sig_table:

                # Construct an MI key by concatenating the signature size with a
                #   partial signature.  Mark all the MI keys that were not marked before.
                for ps in partial_sigs[ov_size]:
                    mi_token = str(sig_size) + "-" + ps
                    markers.setdefault(mi_token, cluster_id)

        # Return the selected cluster ID.
        return cluster_id


def main(name, id='0'):
    mqtt.MQTTClient(name, init=init, funct=process_request, id=id, type=mqtt.CLUSTER)

if __name__ == '__main__':
    # main('mimosa_clustering_pca')
    c = lambda: None
    init(c)
    x, res = process_request(c, ds.Cluster(token='theffballers'))
    print(res)

