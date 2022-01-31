import random
import statistics



def split_lines(input, seed, output1, output2):
    random.seed(seed)
    f1 = open(output1, 'w')
    f2 = open(output2, 'w')

    for line in open(input, 'r').readlines():
        f1.writelines(line) if random.randint(1,2) == 1 else f2.writelines(line)
    return


def tokenize_and_split(sms_file):
    dico = {}
    arr1 = []
    arr2 = []
    cnt  = 0
    for line in open(sms_file, 'r').readlines():
        list = line.split()
        tmp1 = []
        tmp2 = []
        for count, token in enumerate(list) : 
            if count>0 :
                if token not in dico : 
                    dico[token] = cnt
                    cnt+=1
                tmp1.append(dico.get(token)) if list[0]=='spam' else tmp2.append(dico.get(token))
        if tmp1 : arr1.append(tmp1)
        if tmp2 : arr2.append(tmp2)
    return(dico,arr1,arr2)

def compute_frequencies(num_words, documents):
    output = [0]*num_words
    size = len(documents)
    for array in documents :
        for i in set(array) :
            output[i]+=1
    output = [elt/size for elt in output]
    return output


def flatten(t):
    return [item for sublist in t for item in sublist]

def naive_bayes_train(sms_file):
    input = tokenize_and_split(sms_file)
    spam_array = input[1]
    spam_len, total_len = len(spam_array), len(input[1])+len(input[2])
    spam_ratio = 0 if total_len ==0 else spam_len/total_len
    words = input[0]
    num_words = len(words)
    concatList = input[1],input[2]
    total_words = flatten(concatList)
    spam_list = compute_frequencies(num_words, input[1])
    total_list = compute_frequencies(num_words, total_words)
    naive_bayes = 0 if total_list == 0 else [spam_list[i]/total_list[i] for i,elt in enumerate(spam_list)]
    return spam_ratio, words, naive_bayes

def naive_bayes_predict(spam_ratio, words, spamicity, sms):
    products= 1
    tokens = sms.split()
    lst = list(set(tokens))
    for i in lst : 
        if i in words :
            products *= spamicity[words.get(i)]
    return spam_ratio*products


def naive_bayes_eval(test_sms_file, f):
    recall, precision = 0, 0
    truePositive, falsePositive, falseNegative, trueNegative = 0, 0, 0, 0
    for line in open(test_sms_file, 'r').readlines():
        tokens =line.split()
        eval = f("".join(tokens[1:]))
        if eval == 1 and tokens[0] == 'spam' : 
            truePositive+=1
        if eval != 1 and tokens[0] == 'spam' :
            falseNegative+=1
        if eval == 1 and tokens[0] != 'spam' :
            falsePositive+=1
        if eval != 1 and tokens[0] != 'spam' :
            trueNegative +=1
    recall    = 1 if truePositive+falseNegative==0 else truePositive/(truePositive+falseNegative)
    precision = 1 if truePositive+falsePositive==0 else truePositive/(truePositive+falsePositive)
    return (recall, precision)



def alwaysTrue(x):
    return 1
def alwaysFalse(x):
    return 1

spam_ratio, words, spamicity = naive_bayes_train("SMSSpamCollection")
_recall_      = naive_bayes_eval("SMSSpamCollection", alwaysTrue)[0]
_precision_   = naive_bayes_eval("SMSSpamCollection", alwaysTrue)[1]

def classify_spam(sms):
    return naive_bayes_predict(spam_ratio, words, spamicity, sms) > 0.5

def classify_spam_precision(sms):
    return naive_bayes_predict(spam_ratio, words, spamicity, sms) > 0.9

def classify_spam_recall(sms):
    return naive_bayes_predict(spam_ratio, words, spamicity, sms) > 0.9