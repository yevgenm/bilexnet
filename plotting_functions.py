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
