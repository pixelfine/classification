from operator import index
from numpy import append, nan
import numpy as np
import sklearn
from sklearn import cluster
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
import pandas as pd
from collections import Counter
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import KERNEL_PARAMS, cosine_similarity
from collections import defaultdict
import random as rand
import math


data=open('jester_jokes.txt','r').readlines()
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')


def tfidf(list):
    return vectorizer.fit_transform(list)


x = tfidf(data)
#sim = cosine_similarity(x, x[86])
#print(data[0])
#print(data[86])
#print(x)
#print(sim)

def flatten(t):
    return [item for sublist in t for item in sublist]


def min_index(sim_lst, lst):
    min, min_i = 9999, 9999
    for i,elt in enumerate(lst) : 
        if sim_lst[elt] < min :
            min = sim_lst[elt]
            min_i = i
    return min_i


def most_similar(tfidf, item_index, k):
    sim_lst = flatten(cosine_similarity(tfidf, tfidf[item_index]))
    k_most_sim = []
    size = 0
    for i,elt in enumerate(sim_lst) :
        if (i == item_index) : continue
        if size>=k : 
            min_id = min_index(sim_lst, k_most_sim)
            if sim_lst[k_most_sim[min_id]] < elt: 
                del k_most_sim[min_id]
                size-=1
        if size<k :
            k_most_sim.append(i)
            size +=1
    return k_most_sim

#print(most_similar(x, 0, 5))

#Pas besoin de num_jokes sur le dic
def read_ratings(filename, num_jokes):
    csv_data=open(filename,'r').readlines()
    lst = [dict() for x in range(int(((csv_data[-1]).split(','))[0])+1) ]
    for i,line in enumerate(csv_data) : 
        tokens =line.split(',')
        dic = lst[int(tokens[0])]
        dic[int(tokens[1])] =  float(tokens[2])
    return lst

#r=read_ratings('jester_ratings.csv', 150)
#print(r[0])
#print(sum([r[0][x] for x in r[0]]))

#connues  : rating (pair) 
#inconnues: sim_matrix(impair)

def content_recommend(similarity_matrix, user_ratings, k):
    evens, selections = [x for x in user_ratings if x%2==0], []
    for i in range (1, len(similarity_matrix), 2) :
        if i not in user_ratings : 
            sr_sum, s_sum = 0, 0
            for rated in evens :
                #print(i, "\n------\n",similarity_matrix[i][rated], "\n------\n",rated, "\n------\n",user_ratings[rated], "\n------\n") 
                sr_sum += similarity_matrix[i][rated] * user_ratings[rated]
                s_sum  += similarity_matrix[i][rated]
            selections.append( (i, sr_sum/s_sum) )
    selections.sort(key=lambda x:x[1])
    index_lst, rating_lst = (zip(*selections[-k:]))
    return index_lst

        

#ratings = read_ratings('jester_ratings.csv', 150)
#print(content_recommend(cosine_similarity(x), ratings[0], 5))

#cosim = cosine_similarity(x)
#cosim0= cosine_similarity(x, x[0])
#print(cosim, "\n---------------------\n")
#print(cosim[0], "\n---------------------\n")
#print(cosim0, "\n---------------------\n")



def selectN(ratings, nb_to_select, min_size):
    selected = []
    for i, user in enumerate(ratings) : 
        if len(user) >=min_size :
            selected.append(i)
    rand.shuffle(selected)
    return selected[:nb_to_select]


def med(jokes, max_jokes) : 
    count, sum = 0, 0
    for k in jokes : 
        count+=1
        sum+=jokes[k]
    med = sum/count
    for i in range(max_jokes) : 
        jokes[i] = jokes.get(i, med)
    return jokes

def fill_med(users, ratings):
    selection = []
    for i in users : 
        user = ratings[i]
        selection.append(med(user, 150))
    return selection


def to_matrice_users(selected_users, user_cible, joke_cible):
    matrice = []
    for users in selected_users : 
        jokes = []
        for k in users :
            if k in user_cible and (k != joke_cible):
                if users[k] ==0 or users[k] == None : 
                    jokes.append(0.1)
                else : jokes.append(users[k]  )
        matrice.append(jokes)
    return matrice


def moy_matrice(x, size):
    matrice = [[0 for i in range(size)] for j in range(size)]
    for i, a in enumerate(x) : 
        sum = 0
        for j, b in enumerate(a) :
            if (not math.isnan(b)) : sum+=b
        for j, b in enumerate(a) : 
            if math.isnan(b) :
                matrice[i][j] = sum/size
            else : matrice[i][j] = b
    return matrice

def new_jokes(user, max_jokes) :
    lst = []
    for i in range(max_jokes):
        if i not in user : lst.append(i)
    return lst


def randLst(size) : 
    lst = []
    for i in range(size) : 
        lst.append(i)
    rand.shuffle(lst)
    return lst


def collaborative_recommend(ratings, user_ratings, k):
    if len(user_ratings) < 1 : 
        return randLst(150)[:k]
    users = selectN( ratings, 100, 100)
    users_m = fill_med(users, ratings)
    users_m.append(user_ratings)
    recom_list = new_jokes(user_ratings, 150)
    classement_list = []
    for i in recom_list : 
        m = to_matrice_users(users_m, user_ratings, i)
        C = np.corrcoef(m)
        c = moy_matrice(C, len(C))
        score = 0
        count=0
        for j in range(len(c)-1):
            score+= c[j][-1] * users_m[j][i]
            count+=1
        pair =  (i, score/count)
        classement_list.append( pair )
    classement_list.sort(key=lambda x:x[1])
    index_lst, rating_lst = (zip(*classement_list[-k:]))
    return index_lst
    

ratings = read_ratings('jester_ratings.csv', 150)
#print(collaborative_recommend(ratings, ratings[1227], 5))