# https://ragrawal.wordpress.com/2013/01/18/comparing-ranked-list/
# https://github.com/ragrawal/measures/blob/master/measures/rankedlist/RBO.py
import numpy as np
import csv

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
        l[idx] = (w1, w2, weight*c/connections[w1])
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
        d[(w1, w2)] = weight*c/connections[w1]
    return d

def normalize_dict(d, target=1.0):
    #Normalizes dictionary values to 1.
    if target == 0:
        return {}
    raw = sum(d.values())
    if raw == 0: return{}
    factor = target/raw
    return {key:value*factor for key,value in d.items()}

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

def levLoader(theta):
    # Reads the files with orthographic similarities and returns a dictionary {(word1, word2): sim}
    fil = open("./levdist.csv" ,"r")
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
