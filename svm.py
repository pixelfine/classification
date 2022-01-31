import math
import random
import numpy as np
from scipy.sparse import data
import sklearn
from sklearn.model_selection import KFold


def read_data(filename):
    sick_list, data_list, data =[], [], open(filename,'r').readlines()
    for line in data : 
        l = line.replace('\n', '')
        tokens = l.split(",")
        sick_list.append(True if tokens[1]=='M' else False)
        data_list.append([float(e) for e in tokens[2:]])
    return data_list, sick_list

#print(read_data('tmp.txt'))


def simple_distance2(data1, data2):
    sum= np.sum(   (np.array(data1) - np.array(data2) )**2  )
    return math.sqrt(sum)
#print(simple_distance2([1.0, 0.4, -0.3, 0.15], [0.1, 4.2, 0.0, -1]))

def simple_distance(data1, data2):
    sum = 0
    for i,e in enumerate(data2) :
        sum+=((data1[i]-data2[i])**2)
    return math.sqrt(sum)

#print(simple_distance([1.0, 0.4, -0.3, 0.15], [0.1, 4.2, 0.0, -1]))


def k_nearest_neighbors(x, points, dist_function, k):
    if k > len(points) : return []
    selections = []
    for i,elt in enumerate(points) : 
        pair = i, dist_function(x, elt)
        selections.append(pair) 
    if len(selections)==0 : return []
    selections.sort(key=lambda x:x[1])
    index_lst, score_lst = (zip(*selections[:k]))
    return index_lst

#print(k_nearest_neighbors([1.2, -0.3, 3.4],[[2.3, 1.0, 0.5], [1.1, 3.2, 0.9], [0.2, 0.1, 0.23], [4.1, 1.9, 4.0]],simple_distance, 2))



def split_lines(input, seed, output1, output2):
    random.seed(seed)
    f1 = open(output1, 'w')
    f2 = open(output2, 'w')

    for line in open(input, 'r').readlines():
        f1.writelines(line) if random.randint(1,2) == 1 else f2.writelines(line)
    return
#split_lines('wdbc.data', 5, 'train', 'test')


def is_cancerous_knn(x, train_x, train_y, dist_function, k):
    cancerous = k_nearest_neighbors(x, train_x, dist_function, k)
    nb_true = 0
    for index in cancerous : 
        nb_true+= train_y[index] == True 
    return (nb_true>=k/2)
#print(is_cancerous_knn([1.2, -0.3, 3.4],[[2.3, 1.0, 0.5], [1.1, 3.2, 0.9], [0.2, 0.1, 0.23], [4.1, 1.9, 4.0]],[True, False, True, False], simple_distance, 2))
#print(is_cancerous_knn([1.2, -0.3, 3.4],[[2.3, 1.0, 0.5], [1.1, 3.2, 0.9], [0.2, 0.1, 0.23], [4.1, 1.9, 4.0]],[False, False, True, False], simple_distance, 2))
#print(is_cancerous_knn([1.2, -0.3, 3.4],[[2.3, 1.0, 0.5], [1.1, 3.2, 0.9], [0.2, 0.1, 0.23], [4.1, 1.9, 4.0]],[False, False, False, False], simple_distance, 2))

def eval_cancer_classifier(test_x, test_y, classifier):
    error, tot = 0, len(test_y)
    results = classifier(test_x)
    for i,result in results : 
        if test_y[i] != result : error+=1
    return error/tot




def cross_validation_KFold(train_x, train_y, untrained_classifier, k, n):
    kf = KFold(n_splits=k)
    ratio_lst = []
    for train_index, test_index in kf.split(train_x):
        tx =  [train_x[i] for i in train_index]
        ty =  [train_y[i] for i in train_index]
        error = 0
        for x in [train_x[i] for i in test_index] :
            if n == -1 : 
                error+=1 if(untrained_classifier (tx, ty, x) == False) else 0 
            else :
                error+=1 if(untrained_classifier (tx, ty, n, x) == False) else 0 
        ratio_lst.append(error/len(test_index))
    return np.mean(ratio_lst)


def cross_validation(train_x, train_y, untrained_classifier):
    return cross_validation_KFold(train_x, train_y, untrained_classifier, 5, -1)


test_data = read_data('test')
train_data = read_data('train')
tx = train_data[0]
ty = train_data[1]
untrained_classifier = lambda train_x, train_y, x:is_cancerous_knn(x, train_x, train_y, simple_distance, 5)
#print(cross_validation(tx, ty, untrained_classifier))



def sampled_range(mini, maxi, num):
    if not num: return []
    lmini = math.log(mini)
    lmaxi = math.log(maxi)
    ldelta = (lmaxi - lmini) / (num - 1)
    out = [x for x in set([int(math.exp(lmini + i * ldelta)) for i in range(num)])]
    out.sort()
    return out


def find_best_k(train_x, train_y, untrained_classifier_for_k):
    min_idx, min = 0, math.inf
    for i in sampled_range(2, len(train_x), 10):
        score = cross_validation_KFold(train_x, train_y, untrained_classifier_for_k, 10, i) 
        if score < min : min_idx, min = i, score
    return min_idx
    


#untrained_classifier_for_k = lambda train_x, train_y, k, x: is_cancerous_knn(x, train_x, train_y, simple_distance, k)
#best_k = find_best_k(tx, ty, untrained_classifier_for_k)
#print(best_k)
#print(cross_validation_KFold(tx, ty, untrained_classifier_for_k, 10, best_k)) 


def get_weighted_dist_function(train_x, train_y):
    transpose = [[train_x[i][j] for i in range(len(train_x))] for j in range(len(train_x[0]))]
    var_lst = [np.var(x) for x in transpose]
    return lambda x, y: simple_distance(x,y)/ ((sum(var_lst))/(len(var_lst)))

#get_weighted_dist_function(tx, ty)




from sklearn import svm
#from skopt import BayesSearchCV
#def bestSVM(train_x, train_y): 
#    opt = BayesSearchCV(svm.SVC(),{'C': (1e-6, 1e+6, 'log-uniform'),'kernel' : ['rbf'],},n_iter=32,cv=3)
#    opt.fit(train_x, train_y)
#    print("\nOPT\n================================\n",opt,"\n==========================================\n")
#    print("val. score: %s" % opt.best_score_)
#    print(opt.best_params_)
#bestSVM(tx, ty)
# val. score: 0.9523809523809524
# OrderedDict([('C', 14989.193964801012), ('kernel', 'rbf')])

def svm_classify(train_x, train_y, X):
    classifier = svm.SVC(C=14989.193964801012, kernel='rbf')
    classifier.fit(train_x, train_y)
    return classifier.predict(X)
#print(svm_classify(tx, ty, test_data[0]))



def simple_distance_kernel(X, Y, K=simple_distance2):
    X_a, Y_a = np.array(X), np.array(Y)
    matrice = np.zeros((X_a.shape[0], Y_a.shape[0]))
    for i, x in enumerate(X_a) : 
        for j, y in enumerate(Y_a) : 
            matrice[i,j] = K(x,y)
    return matrice

def svm_classify_dist(train_x, train_y, distance_function, X):
    classifier = svm.SVC(C=14989.193964801012, kernel=distance_function)
    classifier.fit(train_x, train_y)
    return classifier.predict(X)
#print(svm_classify_dist(tx, ty, simple_distance_kernel, test_data[0]))
