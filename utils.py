# https://ragrawal.wordpress.com/2013/01/18/comparing-ranked-list/
# https://github.com/ragrawal/measures/blob/master/measures/rankedlist/RBO.py
import numpy as np
import csv
import math
import operator
import pandas



def read_frequencies(fn, lang):
    # Reads the file with Dutch word frequencies.
    freq_dict = {}
    with open(fn) as f:
        reader = csv.reader(f, skipinitialspace=True, quotechar=None, delimiter="\t")
        next(reader)
        for row in reader:
            word = row[0].lower()+lang
            freq = int(row[1])
            if word not in freq_dict: freq_dict[word] = freq
            freq_dict[word] += freq
    return freq_dict



def read_alignment_frequencies(fn="en-nl.refined.dic.lemmas.csv"):
   

    freq = {}
    with open(fn) as f:
        reader = csv.reader(f, skipinitialspace=True, quotechar=None, delimiter="\t")
        for row in reader:
            #print(row)
            words = row[0].split(',')
            #print(words)
            if len(words)==3:
                eng = words[1]
                dutch = words[2]
                freq[(eng.lower(), dutch.lower())]=words[0]
                
    return freq
            
    
    
    
    
def read_dict(fn):
    # Reads an English-Dutch dictionary from CSV.
    dic = {}
    with open(fn) as fin:
        reader = csv.reader(fin, skipinitialspace=True, quotechar=None)
        next(reader)
        for row in reader:
            source = row[0]+":EN"
            transl = row[1]+":NL"
            if source not in dic: dic[source] = []
            dic[source].append(transl)
    return (dic)

def normalize_tuple_list(l, weight):
    # Given a list of tuples (word1, word2, weight), normalizes all weights, so that all edges outgoing from each word add up to the value of weight.
    if weight == 0:
        return []
    connections = {}
    for (w1, w2, c) in l:
        if w1 not in connections:
            connections[w1] = 0
        connections[w1] += c
    for idx in range(len(l)):
        w1 = l[idx][0]
        w2 = l[idx][1]
        c = l[idx][2]
        l[idx] = (w1, w2, weight*c/float(connections[w1]))
    return l

def normalize_tuple_dict(d, weight):
    # Given a dictionary {(word1, word2): weight}, normalizes all weights, so that all edges outgoing from each word add up to the value of weight.
    if weight == 0:
        return {}
    connections = {}
    for (w1, w2), c in d.items():
        if w1 not in connections:
            connections[w1] = 0
        connections[w1] += c
    for (w1, w2), c in d.items():
        d[(w1, w2)] = weight*c/float(connections[w1])
    return d

def normalize_dict(d, target=1.0):
    # Normalizes dictionary values to 1 (or another target value).
    raw = math.fsum(d.values())
    if raw == 0: return {}
    factor = target / raw
    for k in d:
        d[k] = d[k]*factor
    key_for_max = max(d.items(), key=operator.itemgetter(1))[0]
    diff = 1.0 - math.fsum(d.values())
    #print("discrepancy = " + str(diff))
    d[key_for_max] += diff
    return d

def invert_dict(d):
    # Inverts a dictionary of {string:list} pairs.
    newdict = {}
    for k in d:
        for v in d[k]:
            newdict.setdefault(v, []).append(k)
    return newdict

def filter_test_list(G, test_list):
    # Filters out test words that are absent in the big graph.
    test_list_cleaned = [w for w in test_list if w in G.vs['name']]
    test_list_cleaned = [w for w in test_list_cleaned if G.incident(w)]
    return(test_list_cleaned)

def levLoader(theta, fn):
    # Reads the files with orthographic similarities and returns a dictionary {(word1, word2): sim}
    fil = open(fn ,"r")
    l = fil.readlines()
    l = l[1:]
    d = {}
    for lin in l:
        w = lin.split(",")
        #print(w)
        sim = float(w[2].strip('\n').strip('\r'))
        if sim >= theta:
            d[(w[0]+":EN", w[1]+":NL")] = sim
    return d

def syntLoader(fn, words):
    # Reads the files with syntagmatic cooccurence information and returns a dictionary {(word1, word2): sim}
    df = pandas.read_csv(fn, sep="\t", na_values="", keep_default_na=False)
    df["w1"] = df["w1"] + ":EN"
    df["w2"] = df["w2"] + ":EN"
    df = df[(df['w1'].isin(words)) & (df['w2'].isin(words))]
    d = df.set_index(["w1","w2"]).to_dict()["score"]
    return d

def get_rbo(l1, l2, p=0.9):
    """
        Calculates Ranked Biased Overlap (RBO) score.
        l1 -- Ranked List 1
        l2 -- Ranked List 2
    """
    if l1 == None: l1 = []
    if l2 == None: l2 = []

    sl, ll = sorted([(len(l1), l1), (len(l2), l2)])
    s, S = sl
    l, L = ll
    if s == 0: return 0

    # Calculate the overlaps at ranks 1 through l
    # (the longer of the two lists)
    ss = set([])  # contains elements from the smaller list till depth i
    ls = set([])  # contains elements from the longer list till depth i
    x_d = {0: 0}
    sum1 = 0.0
    for i in range(l):
        x = L[i]
        y = S[i] if i < s else None
        d = i + 1

        # if two elements are same then
        # we don't need to add to either of the set
        if x == y:
            x_d[d] = x_d[d - 1] + 1.0
        # else add items to respective list
        # and calculate overlap
        else:
            ls.add(x)
            if y != None: ss.add(y)
            x_d[d] = x_d[d - 1] + (1.0 if x in ss else 0.0) + (1.0 if y in ls else 0.0)
        # calculate average overlap
        sum1 += x_d[d] / d * pow(p, d)

    sum2 = 0.0
    for i in range(l - s):
        d = s + i + 1
        sum2 += x_d[d] * (d - s) / (d * s) * pow(p, d)

    sum3 = ((x_d[l] - x_d[s]) / l + x_d[s] / s) * pow(p, l)

    # Equation 32
    rbo_ext = (1 - p) / p * (sum1 + sum2) + sum3

    return rbo_ext

def get_rbd(l1, l2):
    return(1-get_rbo(l1,l2))



