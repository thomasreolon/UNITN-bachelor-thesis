import pymongo
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import random
import numpy as np

host = 'localhost'
port = 27017
max_reconnect = 3

plot_type = 'clu'


# accounts to plot
tokens = ['theffballers', 'patmcgrathreal']

def myPlot(dat:dict):
    pass


genere = [
    [49,  3],
    [4,   44]
]


eta = [
    [35, 39],    [36, 39],    [39, 35],    [30, 26],    [27, 34],    [34, 36],    [30, 40],    [33, 22],    [27, 32],    [29, 27],    [34, 40],    [31, 38],    [50, 38],    [23, 29],    [29, 37],    [27, 22],    [30, 37],    [32, 35],    [25, 32],    [34, 32],    [40, 37],    [25, 22],    [29, 17],    [32, 34],    [45, 42],    [34, 34],    [32, 27],    [20, 24],    [35, 37],    [27, 25],    [17, 24],    [29, 24],    [32, 38],    [44, 45],    [61, 44],    [27, 36],    [34, 29],    [35, 23],    [22, 25],    [43, 36],    [29, 29],    [31, 23],    [30, 27],    [33, 17],    [45, 38],    [19, 21],    [32, 35],    [40, 35],    [45, 40],    [24, 25],    [40, 20],    [25, 26],    [29, 30],    [40, 39],    [40, 36],    [50, 37],    [26, 31],    [45, 41],    [20, 17],    [28, 26],    [17, 20],    [15, 19],    [16, 16],    [14, 17],    [23, 33],    [18, 17],    [17, 17],    [15, 30],    [21, 23],    [23, 20],    [22, 24],    [20, 18],    [16, 21],    [16, 26],    [25, 23],    [24, 20],    [20, 21],    [20, 23],    [35, 38],    [40, 31],    [26, 17],    [30, 26],    [30, 30],    [40, 43],    [40, 29],    [23, 21],    [29, 32],    [24, 25],    [40, 37],    [34, 39]
]


if __name__ == '__main__' and plot_type=='interests_avg':
    import pymongo
    import random
    users = pymongo.MongoClient(host, port)['personas']['users']
    import mycommons.data_structure as ds

    average = ds.User(img_topics={}, interests={})
    tot_img, tot_tap = 0, 0
    for user in users.find({}):
        to_add = [(average.interests, user['interests']), (average.img_topics, user['img_topics'])]
        for avg, us in to_add:
            if not us: continue
            keys = list(us.keys())
            if 'War' not in keys and 'AI' not in keys: continue
            if 'War' in keys:
                tot_img+=1
            else:
                tot_tap+=1

            for k in keys:
                if k in avg:
                    avg[k] += us[k]
                else:
                    avg[k] = us[k]
        if random.random()>0.95:
            print(tot_tap, tot_img, '|',average)
    for k, v in average.img_topics.items():
        average.img_topics[k] = v / tot_img
    for k, v in average.interests.items():
        average.interests[k] = v/tot_tap

    users.insert_one({
        '_id':0,
        'interests':average.interests,
        'img_topics':average.img_topics
    })



if __name__ == '__main__' and plot_type == 'interests_avg2':
    import pymongo
    import random
    users = pymongo.MongoClient(host, port)['personas']['users']
    import mycommons.data_structure as ds

    average = users.find_one({'_id':0})
    for user in users.find({}):
        if user['_id']==0 or ('averaged' in user and user['averaged']):continue
        if user['interests'] and 'AI' in user['interests'] and user['img_topics'] and 'War' in user['img_topics']:
            for k in user['interests']:
                user['interests'][k] += 1-average['interests'][k]
            for k in user['img_topics']:
                user['img_topics'][k] += 1-average['img_topics'][k]
            user['averaged'] = True
            users.update_one({'_id':user['_id']}, {"$set": user})
            print('updated')
        else:
            users.delete_one({'_id':user['_id']})
            print('del')





myPlot = None

##################################### plot functions ##########################################

def plotPCA(X, Y, features):
    X = np.array(X)
    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    principal_comp = pca.fit_transform(X)

    principalDf = pd.DataFrame(data=principal_comp
                               , columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, pd.DataFrame(Y, columns=['target'])], axis=1)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    colors = ['r', 'g', 'b', 'y']
    targets = set(Y)
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=20)
    ax.legend(targets)
    ax.grid()
    plt.show()

    plt.matshow(pca.components_, cmap='viridis')
    plt.yticks([0, 1], ['1st Comp', '2nd Comp'], fontsize=10)
    plt.colorbar()
    plt.xticks(range(len(X[0])), features, rotation=65, ha='left')
    plt.tight_layout()
    plt.show()  #

##################################################

def plot2D(X,Y,features, invert_axes=False):
    x,y = [],[]
    a,b = 0,1
    if invert_axes:
        a,b = 1,0
    for point in X:
        x.append(point[a])
        y.append(point[b])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(features[a], fontsize=15)
    ax.set_ylabel(features[b], fontsize=15)
    ax.set_title('2D plot', fontsize=20)
    colors = {comp:c for comp,c in zip(set(Y), ['r', 'g', 'b', 'y'])}

    for p1, p2, cmp in zip(x,y, Y):
        ax.scatter(p1, p2, c=colors[cmp], s=50)

    ax.legend(tuple(set(Y)), loc=1, title="tokens")
    ax.grid()
    plt.show()




if __name__=='__main__' and plot_type=='gender':
    import seaborn as sn
    import pandas as pd


    df_cm = pd.DataFrame(genere, index=['act. Male','act. Female'],
                         columns=['pred. Male','pred. Female'])
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, cmap='Blues')

    plt.show()
    print('done', genere)

def get_pca(all_cl, token=None):
    import pymongo
    pca = PCA(n_components=2, svd_solver='arpack') # gli altri solver non vanno!! fortran schifoh
    scaler = StandardScaler()
    X, tokens = [], set()
    yo = pymongo.MongoClient('localhost', 27017)['personas']['users']
    query = {} if not token else {'token':token}

    # new_all = {}
    # for file in all_cl:
    #     if file['token'] in new_all:
    #         f, num = new_all[file['token']]
    #         mm =  sum([len(x) for x in file['cluster']])
    #         if mm>num:
    #             new_all[file['token']] = (file, sum([len(x) for x in file['cluster']]))
    #     else:
    #         new_all[file['token']] = (file, sum([len(x) for x in file['cluster']]))
    # new_all = [y[0] for x, y in new_all.items()]
    #
    # for file in new_all:
    #     if file['token'] in tokens:
    #         continue
    #
    #     tokens.add(file['token'])
    #     clusters = file['cluster']
    #     for cl in clusters:
    for user in yo.find(query):
        if not 'age' in user or not user['age'] or user['age'] < 0 or 'AI' not in user['interests']: continue
        point = ds.Insights(user)
        # del point.ocean
        # del point.needs
        # del point.nlu
        del point.mbti
        # if 'und' in point.language:
        #     del point.language['und']
        del point.language
        if len(point.interests)>0:
            X.append(point.get_point())
            xx = len(point.get_point())
            if not xx in tokens:
                tokens.add(xx)
                print(point.get_point())


    scaler = scaler.fit(X)
    pca = pca.fit(X)

    return pca, scaler



#######################################################################################################################
if __name__=='__main__' and plot_type=='clu':
    from os import listdir
    from os.path import isfile, join
    import json
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.covariance import EllipticEnvelope

    import mycommons.data_structure as ds
    all_cl = []

    onlyfiles = [join('/tmp', f) for f in listdir('/tmp') if isfile(join('/tmp', f)) and 'clsave' in f]
    for f in onlyfiles:
        with open(f, 'r') as fin:
            clusters = json.load(fin)
        all_cl.append(clusters)

    #pca, scaler = get_pca(all_cl)

    # pca how
    # plt.matshow(pca.components_, cmap='viridis')
    # plt.yticks([0, 1], ['1st Comp', '2nd Comp'], fontsize=10)
    # plt.colorbar()
    # features = [
    #     'age', 'is male', 'animals', 'music', 'health', 'war', 'clothing', 'sports', 'drinks', 'rich', 'cosplay', 'office', 'travel', 'family', 'baby',
    #     'it', 'en', 'fr', 'de', 'sp', 'un',
    #     'basketball', 'video games', 'food', 'american football', 'dance', 'basketball', 'tennis', 'religion', 'biking', 'tech', 'academics', 'activism', 'animalsNLU', 'cars', 'animation', 'soccer', 'nature', 'AI', 'design', 'politics'
    # ]
    # plt.xticks(range(len(features)), features, rotation=65, ha='left')
    # plt.tight_layout()
    # plt.show()

    pcas  ={x['token']:None for x in all_cl}
    colors = [[1, 0, 0], [1, 1, 0], [0, 1, 0],  [0, 1, 1], [0, 0, 1], [1, 0, 1]]

    # to just remove the scatter plot and keep the limits

    for file in reversed(all_cl):
        X = []
        if not pcas[file['token']]:
            print(file['token'])
            pcas[file['token']] = get_pca(all_cl, file['token'])
            pca, scaler = pcas[file['token']]

            plt.matshow(pca.components_, cmap='viridis')
            plt.yticks([0, 1], ['1st Comp', '2nd Comp'], fontsize=10)
            plt.colorbar()
            features = [
                'age', 'is male', 'animals', 'music', 'health', 'war', 'clothing', 'sports', 'drinks', 'rich', 'cosplay',
                'office', 'travel', 'family', 'baby',
#                'it', 'en', 'fr', 'de', 'sp', 'un',
                'basketball', 'video games', 'food', 'american football', 'dance', 'basketball', 'tennis', 'religion',
                'biking', 'tech', 'academics', 'activism', 'animalsNLU', 'cars', 'animation', 'soccer', 'nature', 'AI',
                'design', 'politics'
            ]
            plt.xticks(range(len(features)), features, rotation=65, ha='left')
            plt.tight_layout()
            plt.title(file['token'])
            plt.show()
        pca, scaler = pcas[file['token']]

        for cl, col in zip(file['cluster'], colors):
            for user in cl:
                if not 'age' in user or not user['age'] or user['age'] < 0 or 'AI' not in user['interests']: continue

                point = ds.Insights(user)
                del point.mbti
                del point.language
                point = scaler.transform([point.get_point()])[0]
                point = pca.transform([point])[0]
                X.append((point[0], point[1], col))

        clf_o = EllipticEnvelope(contamination=0.1)
        clf_o.fit([[x,y] for x,y,z in X])
        pred = clf_o.predict([[x, y] for x, y, z in X])
        Y = np.array([x for x,pr in zip(X,pred) if pr>0])
        X = [[x,y] for x,y,z in Y]
        m, M = np.min(X, axis=0), np.max(X, axis=0)

        X = (X - m) * 100 / (M - m) * (1 - pca.explained_variance_ratio_)
        for i, x in enumerate(X):
            Y[i] = [x[0],x[1],Y[i][2]]
        X = Y


        plt.scatter([x[0] for x in X], [x[1] for x in X], c=[x[2] for x in X])
        plt.title(file['token']+ ': '+file['type'])
        plt.show()
        plt.clf()

if plot_type=='age' and __name__=='__main__':
    from pylab import *
    x = np.array([p[0] for p in eta])
    y = np.array([p[1] for p in eta])
    y = [(a+b)/2 for a,b in zip(x,y)]

    lineStart, lineEnd = 10, 55
    coef = np.polyfit(x, y, 1)
    poly1d_fn = np.poly1d(coef)


    #plt.scatter(x, y)#, s=area, c=colors, alpha=0.5)
    plt.plot(x,y, 'bo', x, poly1d_fn(x), '--k')
    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color='r')
    plt.xlim(lineStart, lineEnd)
    plt.ylim(lineStart, lineEnd)
    xlabel('Real Age')
    ylabel('Predicted Age')
    plt.show()

    err = sum(abs(a-b) for a,b in zip(x,y))





if plot_type=='cluster1' and __name__=='__main__':
    # code
    db_users = pymongo.MongoClient(host, port)['personas']['users']
    X = []
    Y = []
    prev = 0
    features = None
    print('-------------------\nvalid users / total users')
    for account in tokens:
        users = db_users.find({'token': account})
        tot = 0
        for user in users:
            tot += 1
            point, dat = None, None
            try:
                if isinstance(user['gender'], str):
                    if user['gender']=='M':
                        gender = 1. + random.random() / 5
                    else:
                        gender = random.random() / 5
                else:
                    gender = float(user['gender'])
                dat = {
                    "age":user['age'],
                    "gen-male":gender,
                    # "O":user['ocean']['O'],
                    # "C":user['ocean']['C'],
                    # "oE":user['ocean']['E'],
                    # "A":user['ocean']['A'],
                    # "N":user['ocean']['N'],
                    # "T":user['mbti']['T'],
                    # "S":user['mbti']['S'],
                    # "E":user['mbti']['E'],
                    # "J":user['mbti']['J'],
                    # "cons":user["needs"]["conservative"],
                    # "expr":user["needs"]["need_expression"],
                    # "love":user["needs"]["need_love"],
                    # "anger":user["needs"]["anger"],
                    # "depr":user["needs"]["depression"],
                    # "stress":user["needs"]["stress"],
                    # "intell":user["needs"]["intellect"],
                    "animal":user["img_topics"]["animals"],
                    "music":user["img_topics"]["music"],
                    "hospital":user["img_topics"]["hospital"],
                    "war":user["img_topics"]["war"],
                    "cloth":user["img_topics"]["clothes"],
                    "sport":user["img_topics"]["sport"],
                    "alchool":user["img_topics"]["alcohol"],
                    "rich":user["img_topics"]["rich"],
                    "mask":user["img_topics"]["costumes"],
                    "office":user["img_topics"]["office"],
                    "holiday":user["img_topics"]["holiday"],
                    "indo":user["nlu"]["categories"]["indoor"],
                    "outd":user["nlu"]["categories"]["outdoor"],
                    "cars":user["nlu"]["categories"]["cars"],
                    "tech":user["nlu"]["categories"]["tech"],
                    "young":user["nlu"]["categories"]["family"],
                    "stran":user["nlu"]["categories"]["politics"],
                }
                if myPlot:
                    dat = myPlot(dat)
                point = [v for k,v in dat.items()]
            except KeyError as e:
                pass
            except TypeError as e:
                pass
            if point and None not in point:
                if not features and dat:
                    features = [k for k in dat]
                X.append(point)
                Y.append(user['token'])
        print(account, ':', len(Y)-prev, '/', tot)
        prev = len(Y)

    # stats on the most distinctive features
    numX = {k:[] for k in set(Y)}
    avgs = {}
    for point, k in zip(X, Y):
        numX[k].append(point)
    for k in set(Y):
        avgs[k] = np.average(numX[k], axis=0)
    print('-------------------\nAVERAGES PER FEATURE')
    for i, feat in enumerate(features):
        text_avg = []
        for k, v in avgs.items():
            text_avg.append(f'<{k[0]}|{v[i]}>')
        print(feat, ':', " ".join(text_avg))


    if len(features)==2:
        plot2D(X,Y,features)
    elif len(features)==3:
        pass
    else:
        plotPCA(X,Y,features)
