import os
import json
import networkx as nx
from networkx.readwrite import json_graph
from collections import Counter
import re


def Graphnx():
    count = 1
    f = open("./Dutch/associationData.csv")
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

def get_lemmatized_english_digraph(new):
    if os.path.isfile("./EAT/EATnew_directed") and new=="no":
        with open("./EAT/EATnew_directed") as jc:
            D = json_graph.node_link_graph(json.load(jc))
    else:
        M = nx.read_pajek("modEAT.net")
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

def test_network(D, test_list, max_depth):
    for w in test_list:
        print("CUE:",w)
        for depth in range(1, max_depth):
            print("\tMAX DEPTH:", depth)
            responses = dict(get_connections(D, {w:1}, depth, current_depth=1))
            responses[w] = 0
            for k, v in sorted(responses.items(), key=lambda x: x[1], reverse=True)[:5]:
                print("\t\t%s\t\t%.3f" % (k, v))


if __name__ == "__main__":
    print('runnning')
    get_lemmatized_english_digraph("yes")
    raw_input()
    enD = get_english_digraph()
    nlD = get_dutch_digraph()

    test = ["skirt", "potato"]
    test_network(enD, test, 4)

    #test = ["oneindig", "eeuwig"]
    #test_network(nlD, test, 4)


