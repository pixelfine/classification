import sklearn
from sklearn import cluster
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
import pandas as pd
from collections import Counter
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

#print(sklearn.__version__)


data_set = 'SMSSpamCollection'

def read_dataset(filename):
    lst = [] 
    for line in open(filename, 'r').readlines():
        tokens =line.split()
        type = 1 if tokens[0]=='spam' else 0
        text = " ".join(tokens[1:])
        lst.append(  (type, text)   )
    return lst


def spam_count(pairs):
    counter = 0
    for elt in pairs :
        counter+=elt[0]
    return counter


vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')

def transform_text(pairs):
    lst, x,y =[], [], []
    for elt in pairs : 
        lst.append(elt[1])
        y.append(elt[0])
    x=vectorizer.fit_transform(lst)
    return (x,y)
    

def test_transform_text() : 
    pairs = read_dataset(data_set)
    (x,y) = transform_text(pairs)
    for a,b in zip(*x.nonzero()):
        print("[%d][%d] = %s" % (a, b, x[a, b]))
#test_transform_text()



def flatten(t):
    return [item for sublist in t for item in sublist]

def smallest_id(lst) : 
    min_v = min(lst)
    return lst.index(min_v)

def kmeans_and_most_common_words(pairs, K, P):
    (x,y) = transform_text(pairs)
    kmean = KMeans(n_clusters=K).fit(x)
    predict = kmean.predict(x)
    cluster_lst = []
    for i in range(K) :
        cluster_lst.append([])

    for (i,elt) in enumerate(x) :
        cluster_lst[predict[i]].append(elt)

    (indices,data) = [], []
    for (i,elt) in enumerate(cluster_lst) :
        i,d = [],[]
        for (j, e) in enumerate(elt) :
            i.append(e.indices)
            d.append(e.data)
        indices.append(flatten(i))
        data.append(flatten(d))
    
    for(i, elt) in enumerate(data) :
        size = len(elt)
        while(size>P) :
            del data[i][smallest_id(elt)]
            del indices[i][smallest_id(elt)]
            size-=1

    cluster_w = []

    names = vectorizer.get_feature_names()
    for tab in indices :
        arr = []
        for idx in tab :
            arr.append(names[idx])
        cluster_w.append(arr)

    print(cluster_w)
    return cluster_w
#kmeans_and_most_common_words(read_dataset(data_set), 3, 4)

def best_k(pairs):
    (x,y) = transform_text(pairs)
    size = x.get_shape()[0]
    scores = []
    for i in range(size-2) : 
        kmean = KMeans(n_clusters=i+2, random_state=20).fit(x)
        predict = kmean.fit_predict(x)
        score = silhouette_score(x, predict)
        scores.append(score)
        print(score)
    print("max = ", scores.index(max(scores)))
    return scores.index(max(scores))

#print(best_k(read_dataset(data_set)))

def agglo_cluster(pairs) : 
    (x,y) = transform_text(pairs)
    clustering = AgglomerativeClustering().fit(x.toarray())
    return x,clustering
#print(  agglo_cluster(read_dataset(data_set))[1] )

def moy_cluster(pairs) : 
    ""

def classify_batch(train_pairs, test):
    """Classification algorithm using clustering, and nearest (euclidian)
    distance: use some clustering algorithm from the previous questions,
    and decide on a way to use them to classify all the messages in “test”.
    Args:
    pairs: see previous questions
    test: a list of messages to classify, e.g. ['Hello world',
    'Did you receive my last call ?', ...]
    Returns:
    A list of the (estimated) types of the messages in “test”: this list
    Will contain as many elements as “test”, each element will be 0 or 1.
    """