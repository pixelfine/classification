# Load EDA Pkgs
from nltk.corpus.reader.chasen import test
from numpy import random
import pandas as pd
import numpy as np
# Load Data Viz Pkgs
import seaborn as sns
# Load Text Cleaning Pkgs
import neattext.functions as nfx
# Load ML Pkgs
# Estimators
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import DistanceMetric

# Transformers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


import pandas as pd  
import numpy as np  
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statistics

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from collections import Counter
from skopt import BayesSearchCV

# Load Dataset -> df is the dataset of training and df_test of testing
df = pd.read_csv("tweets.csv", usecols= ['text', 'target'])

df = df[df['text'].notna()]

def replace_df(df) : 
    df['good'] = df['admiration'] + df['amusement'] +df['approval'] +df['caring'] +df['desire'] +df['curiosity'] +df['joy'] +df['love'] +df['optimism'] +df['gratitude'] +df['relief']
    df['bad'] = df['anger'] + df['annoyance'] +df['disappointment'] +df['disapproval'] +df['disgust'] +df['embarrassment'] +df['fear'] +df['grief'] +df['nervousness'] +df['pride'] +df['remorse'] + df['sadness']
    df['neutral'] = df['confusion'] + df['realization'] +df['neutral'] +df['excitement'] +df['surprise']
    df = df.drop(columns=['admiration','amusement','approval', 'caring','desire', 'curiosity','joy','love','optimism','gratitude','relief',
'confusion','realization','excitement','surprise',
'anger','annoyance','disappointment','disapproval','disgust','embarrassment','fear','grief','nervousness','pride',
'remorse','sadness'])
    return df[~df['text'].str.contains(r'[^\x00-\x7F]')]

#df = replace_df(df)

def verification(data) :
    if data['good'] != 0 : return 0
    if data['bad'] != 0 : return 1
    if data['neutral'] != 0 : return 2
    if data['good'] == 0 and data['bad'] == 0 and data['neutral'] == 0 : return 2
    

#df['class'] = df.apply(lambda row: verification(row), axis=1)
df_classes = df[['text', 'target']]

# fonction de cleaning utilisant des techniques nlp
def nlp_cleaning(df) :
    ret = df['text']
    ret = nfx.remove_userhandles(ret)
    ret = nfx.remove_stopwords(ret)
    ret = nfx.remove_emojis(ret)
    ret = nfx.remove_punctuations(ret)
    ret = nfx.remove_special_characters(ret)
    return ret

# User handles
df_classes['clean_text'] = df_classes.apply(lambda row: nlp_cleaning(row), axis=1)




max_lines = 100000
#df = df_classes[:max_lines].sample(frac=1).reset_index(drop=True)
df = df_classes.sample(frac=1).reset_index(drop=True)
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['clean_text'] , df['target'], test_size=0.1)
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)


vect = TfidfVectorizer()
vect.fit(df_classes['clean_text'])
X_train = vect.transform(Train_X)
X_test = vect.transform(Test_X)


#[0.9387374677879294, 'linear']
Naive = svm.SVC(C=0.9387374677879294, kernel='linear')#naive_bayes.MultinomialNB()
Naive.fit(X_train, Train_Y)


#def mydist (x,y):
    #x_id, y_id   = x.indices, y.indices
    #x_data, y_data = x.data, y.data
    #return 100*(Naive.predict(x)-Naive.predict(y))


def find_bestk() : 
    max, max_k= 0, 0
    count=0
    for i in range(3, 100) : 
        knn_i = KNeighborsClassifier(n_neighbors=i, metric= 'minkowski')
        score = statistics.mean(cross_val_score(knn_i, X_train, Train_Y, cv=5))
        if score > max : 
            max, max_k =score, i
            count=0
        else : count+=1
        if count>3 : return max_k
    print("Best k =", max_k)
    return max_k

#knn = KNeighborsClassifier(n_neighbors=3, metric= 'minkowski')
#clf = knn.fit(X_train, Train_Y)

print("Prediction")
nb_lines_test = 1000
pred_y = Naive.predict(X_test[:nb_lines_test])
report = classification_report(Test_Y[:nb_lines_test], pred_y[:nb_lines_test])
print(report)