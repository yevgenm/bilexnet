import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import re
import pydotplus
from collections import Counter

#np.set_printoptions(threshold=np.nan)



def get_connections(G, responses, depth, current_depth):
    if current_depth > depth:
        return(responses)
    else:
        responses_current_level = dict()
        for word in sorted(responses):
            weight = responses[word]
            new = {r[0]:r[1]['weight'] for r in G.edge[word].items()}
            total = sum(new.values())
            responses_single_word = {k:v/total*weight for k,v in new.items()}
            responses_current_level = dict(sum((Counter(x) for x in [responses_current_level, responses_single_word]), Counter()))
        current_depth += 1
        responses_next_level = get_connections(G,responses_current_level, depth, current_depth)
        final = dict(sum((Counter(x) for x in [responses_current_level, responses_next_level]), Counter()))
        return(final)
    
    
def Graphnx():
    count = 1
    f = open("associationData.csv")
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
            print(w)
            #print(line)
            G.add_nodes_from(w)
            for i in range(3):
                if G.has_edge(w[0], w[i+1]):
                    G[w[0]][w[i+1]]['weight'] = G[w[0]][w[i+1]]['weight'] + 1
                else:
                    G.add_edge(w[0], w[i+1], weight = 1)
    print('EEEENNNNNDDDDDD')
    print(G.number_of_nodes())
    print(G.number_of_edges())
    #nx.draw_networkx(G,pos=nx.spring_layout(G))
    #plt.show()
    return G
    
def Matrix():
    count = 0
    f = open("associationData.csv")
    l = f.readlines()
    l = l[1:]
    
    d = {}
    
    for line in l:
        line = line.strip('\n')
        #print(count)
  
        #print(line)
        w = line.split(";")[2:]
        
        if w[0] not in d:
            d[w[0]] = count
            count = count + 1
            
        if w[1] not in d:
            d[w[1]] = count
            count = count + 1
        if w[2] not in d:
            d[w[2]] = count
            count = count + 1
            
        if w[3] not in d:
            d[w[3]] = count
            count = count + 1
            
    matrix = np.zeros(count*count).reshape(count,count)
    
    
    for line in l:
        line = line.strip('\n')
        #print(count)
    
        #print(line)
        w = line.split(";")[2:]
        ind0 = d[w[0]]
        ind1 = d[w[1]]
        ind2 = d[w[2]]
        ind3 = d[w[3]]
        
        matrix[ind0][ind1] = matrix[ind0][ind1] + 1
        matrix[ind0][ind2] = matrix[ind0][ind2] + 1
        matrix[ind0][ind3] = matrix[ind0][ind3] + 1
        
        
    return matrix



def get_connections(G, responses, depth, current_depth):
    if current_depth > depth:
        return(responses)
    else:
        responses_current_level = dict()
        for word in sorted(responses):
            weight = responses[word]
            new = {r[0]:r[1]['weight'] for r in G.edge[word].items()}
            total = sum(new.values())
            responses_single_word = {k:v/total*weight for k,v in new.items()}
            responses_current_level = dict(sum((Counter(x) for x in [responses_current_level, responses_single_word]), Counter()))
        current_depth += 1
        responses_next_level = get_connections(G,responses_current_level, depth, current_depth)
        final = dict(sum((Counter(x) for x in [responses_current_level, responses_next_level]), Counter()))
        return(final)
    
    
    
if __name__ == "__main__":
    G = Graphnx()
    #test = ["skirt", "potato"]
    test = ["oneindig","eeuwig"]
    
    for w in test:
        print("CUE:",w)
        for depth in range(1,5):
            print("\tMAX DEPTH:", depth)
            responses = dict(get_connections(G, {w:1}, depth, current_depth=1))
            responses[w] = 0
            responses[w] = 0
            for k, v in sorted(responses.items(), key=lambda x: x[1], reverse=True)[:5]:
                print("\t\t%s\t\t%.3f" % (k, v))

    #m = Matrix()
    
    #print(m)
        
        
            
