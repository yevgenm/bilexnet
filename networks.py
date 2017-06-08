import os
import json
import networkx as nx
from networkx.readwrite import json_graph
from collections import Counter
import re
import csv


def Graphnx():
    count = 1
    f = open("./Dutch/associationDataLemmas.csv")
    l = f.readlines()
    l = l[1:]
    G = nx.DiGraph()
    for line in l:
        line = line.strip('\n')
        line = re.sub(r'"', '',line)
        #line = line.strip('"')
        #print(count)
        count = count+1
        if True:#count%10 == 0:
            #print(line)
            w = line.split(";")[2:]
            #print(w)
            #print(line)
            G.add_nodes_from(w)
            for i in range(3):
                if w[i+1] != 'x':
                    if G.has_edge(w[0], w[i+1]):
                        G[w[0]][w[i+1]]['weight'] = G[w[0]][w[i+1]]['weight'] + 1
                    else:
                        G.add_edge(w[0], w[i+1], weight = 1)
    #print('EEEENNNNNDDDDDD')
    print(G.number_of_nodes())
    print(G.number_of_edges())
    #nx.draw_networkx(G,pos=nx.spring_layout(G))
    #plt.show()
    return G


def get_english_digraph():
    if os.path.isfile("./EAT/EATnew_directed"):
        with open("./EAT/EATnew_directed") as jc:
            D = json_graph.node_link_graph(json.load(jc))
    else:
        M = nx.read_pajek("./EAT/EATnew.net")
        D = nx.DiGraph()
        for u, v, data in M.edges_iter(data=True):
            w = data['weight'] if 'weight' in data else 1.0
            if D.has_edge(u, v):
                D[u][v]['weight'] += w
            else:
                D.add_edge(u.lower(), v.lower(), weight=w)
        with open("./EAT/EATnew_directed", 'w') as outfile:
            json.dump(json_graph.node_link_data(D), outfile)
    return D

def get_dutch_digraph():
    if os.path.isfile("./Dutch/associationData_directed"):
        with open("./Dutch/associationData_directed") as jc:
            D = json_graph.node_link_graph(json.load(jc))
    else:
        D = Graphnx()
        with open("./Dutch/associationData_directed", 'w') as outfile:
            json.dump(json_graph.node_link_data(D), outfile)
    return D


def get_connections(G, responses, depth, current_depth):
    if current_depth > depth:
        return(responses)
    else:
        responses_current_level = dict()
        for word in sorted(responses):
            weight = responses[word]
            new = {w:e[2] for e in G.out_edges([word], 'weight') for w in str.split(e[1])}
            total = sum(new.values())
            responses_single_word = {k:v/total*weight for k,v in new.items()}
            responses_current_level = dict(sum((Counter(x) for x in [responses_current_level, responses_single_word]), Counter()))
        current_depth += 1
        responses_next_level = get_connections(G, responses_current_level, depth, current_depth)
        final = dict(sum((Counter(x) for x in [responses_current_level, responses_next_level]), Counter()))
        return(final)

def test_network_print(D, test_list, max_depth, gold={}):
    for w in test_list:
        print("CUE:",w)
        if gold:
            print("\tGOLD")
            for k, v in sorted(gold[w], key=lambda x: x[1], reverse=True)[:5]:
                print("\t\t%s\t\t%.3f" % (k, v))
        for depth in range(1, max_depth):
            print("\tMAX DEPTH:", depth)
            responses = dict(get_connections(D, {w:1}, depth, current_depth=1))
            responses[w] = 0
            for k, v in sorted(responses.items(), key=lambda x: x[1], reverse=True)[:5]:
                print("\t\t%s\t\t%.3f" % (k, v))

def preprocess_word(w):
    if w[0:2] == "vk":
        w = w[2:]
    if w=="geen":
        return None
    if "(" in w:
        w = w[:w.index("(")]
    return(w)

def read_test_file(fn):
    conditions = fn.split("/")[-1].split('.')[0].split('-')
    cue_langs = [c[0] for c in conditions]
    target_langs = [c[1] for c in conditions]
    n_conds = len(conditions)
    resp_dict = {}
    with open(fn) as f:
        test_reader = csv.reader(f, delimiter=",")
        next(test_reader)
        for row in test_reader:
            cue = row[1]
            responses_mixed = row[3:]
            for cond_idx in range(n_conds):
                cue_lang = cue_langs[cond_idx]
                target_lang = target_langs[cond_idx]
                if cue_lang not in resp_dict:
                    resp_dict[cue_lang] = {}
                if target_lang not in resp_dict[cue_lang]:
                    resp_dict[cue_lang][target_lang] = []
                target_responses = [(cue, preprocess_word(responses_mixed[r_idx])) for r_idx in range(len(responses_mixed))
                                    if r_idx%n_conds==cond_idx and preprocess_word(responses_mixed[r_idx])!=None]
                resp_dict[cue_lang][target_lang].extend(target_responses)
    return(resp_dict)


if __name__ == "__main__":
    #enD = get_english_digraph()
    nlD = get_dutch_digraph()

    # DD_DE_test_dict = read_test_file("./vanhell/DD1-DE2-DD3.csv")
    # DD_test_list = DD_DE_test_dict['D']['D']
    # gold_dict = {}
    # for (c, r), f in Counter(DD_test_list).items():
    #     if c not in gold_dict: gold_dict[c] = []
    #     gold_dict[c].append((r,f))
    #
    # for cue in gold_dict.keys():
    #     test_network_print(nlD,[cue],3,gold_dict)


    #test = ["skirt", "potato"]
    #test_network_print(enD, test, 4)

    #test = ["oneindig", "eeuwig"]
    #test_network_print(nlD, test, 4)


