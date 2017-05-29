import os
import json
import networkx as nx
from networkx.readwrite import json_graph
from collections import Counter

# Plotting functions.
# import plotly
# from plotly.graph_objs import *
# def make_annotations(pos, labels, font_size=14, font_color='rgb(25,25,25)'):
#     L = len(pos)
#     if len(labels) != L:
#         raise ValueError('The lists pos and text must have the same len')
#     annotations = Annotations()
#     for k in range(L):
#         annotations.append(
#             Annotation(
#                 text=labels[k],
#                 x=pos[labels[k]][0], y=pos[labels[k]][1] + 0.01,
#                 xref='x1', yref='y1',
#                 font=dict(color=font_color, size=font_size),
#                 showarrow=False)
#         )
#     return annotations
#
#
# def plotGraph(G, labels, pos, fname):
#
#     edge_trace = Scatter(
#         x=[],
#         y=[],
#         line=Line(width=0.5,color='#888'),
#         hoverinfo='none',
#         mode='lines')
#
#     for edge in G.edges():
#         x0, y0 = pos[edge[0]]
#         x1, y1 = pos[edge[1]]
#         edge_trace['x'] += [x0, x1, None]
#         edge_trace['y'] += [y0, y1, None]
#
#     node_trace = Scatter(
#         x=[],
#         y=[],
#         text=[],
#         mode='markers',
#         hoverinfo='text',
#         marker=Marker(
#             showscale=True,
#             colorscale='YIGnBu',
#             reversescale=True,
#             color=[],
#             size=10,
#             colorbar=dict(
#                 thickness=15,
#                 title='Node Connections',
#                 xanchor='left',
#                 titleside='right'
#             ),
#             line=dict(width=2)))
#
#     for node in G.nodes():
#         x, y = pos[node]
#         node_trace['x'].append(x)
#         node_trace['y'].append(y)
#
#     for node, adjacencies in enumerate(G.adjacency_list()):
#         node_trace['marker']['color'].append(len(adjacencies))
#         node_info = '# of connections: '+str(len(adjacencies))
#         node_trace['text'].append(node_info)
#
#     plotly.offline.plot({
#         "data": Data([edge_trace, node_trace]),
#         "layout": Layout(
#                     title=fname,
#                     titlefont=dict(size=16),
#                     showlegend=False,
#                     hovermode='closest',
#                     margin=dict(b=20,l=5,r=5,t=40),
#                     xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
#                     yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False),
#                     annotations=make_annotations(pos, labels))
#     }, filename = fname + '.html')

def get_connections(responses, depth, current_depth):
    if current_depth > depth:
        return(responses)
    else:
        responses_current_level = dict()
        for word in sorted(responses):
            weight = responses[word]
            new = {r[0]:r[1]['weight'] for r in G.edge[word.upper()].items()}
            total = sum(new.values())
            responses_single_word = {k:v/total*weight for k,v in new.items()}
            responses_current_level = dict(sum((Counter(x) for x in [responses_current_level, responses_single_word]), Counter()))
        current_depth += 1
        responses_next_level = get_connections(responses_current_level, depth, current_depth)
        final = dict(sum((Counter(x) for x in [responses_current_level, responses_next_level]), Counter()))
        return(final)

fn="EATnew"

if os.path.isfile("./EAT/"+fn+"_directed") and os.path.isfile("./EAT/"+fn+"_undirected"):
    with open("./EAT/"+fn+"_directed") as jc:
        G = json_graph.node_link_graph(json.load(jc))
else:
    M = nx.read_pajek("./EAT/"+fn+".net")
    D = nx.Graph()
    for u,v,data in M.edges_iter(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if D.has_edge(u,v):
            D[u][v]['weight'] += w
        else:
            D.add_edge(u, v, weight=w)
    with open("./EAT/"+fn+"_directed", 'w') as outfile:
        json.dump(json_graph.node_link_data(D), outfile)
    G = D.to_undirected()
    with open("./EAT/"+fn+"_undirected", 'w') as outfile:
        json.dump(json_graph.node_link_data(G), outfile)

test = ["skirt", "potato"]

for w in test:
    print("CUE:",w)
    for depth in range(1,3):
        print("\tMAX DEPTH:", depth)
        responses = dict(get_connections({w:1}, depth, current_depth=1))
        responses[w.upper()] = 0
        responses[w] = 0
        for k, v in sorted(responses.items(), key=lambda x: x[1], reverse=True)[:5]:
            print("\t\t%s\t\t%.3f" % (k, v))


# Plotting.
# pos=nx.spring_layout(G)
# plotGraph(G, list(G.edge.keys()), pos, "eap_plot")
#
# part_nat = community.best_partition(G)
# mod_nat = community.modularity(part_nat,G)
# values = [part_nat.get(node) for node in G.nodes()]
# nx.draw_spring(G, cmap=plt.get_cmap('jet'), node_color = values, node_size=30, with_labels=True)
# plt.show()
#
