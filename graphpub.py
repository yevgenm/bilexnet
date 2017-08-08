import copy
from igraph import *
import numpy as np
from numpy import inf

def get_cc_directed(G):
    ccs = []
    for v in G.vs:
        neighbors = set(G.neighbors(v, mode=IN))
        #neighbors = set(G.neighbors(v, mode=ALL))
        if len(neighbors) > 1:
            n_edges_potential = len(neighbors) * (len(neighbors)-1)
            #n_edges_potential = 0.5 * len(neighbors) * (len(neighbors) - 1)
            neighbors_edges_all = set([e for n in neighbors for e in G.adjacent(n)])
            neighbors_edges_relevant = [e for e in neighbors_edges_all if G.es[e].tuple[0] in neighbors and G.es[e].tuple[1] in neighbors]
            n_edges_real = len(neighbors_edges_relevant)
            cc = float(n_edges_real)/n_edges_potential
            ccs.append(cc)
    return sum(ccs)/float(len(ccs))


def get_cc(G, directed):
    if directed:
        cc = get_cc_directed(G)
    else:
        cc = G.transitivity_avglocal_undirected()
    return cc


def get_aspl(G):
    aspl_sum = 0.0
    n = 0
    for v in G.vs:
        spl_single = np.array(G.shortest_paths(source=v))
        spl_single[spl_single == inf] = 0
        aspl_sum += np.mean(spl_single)
        n += 1
    aspl = aspl_sum / n
    return aspl


def get_q(G):
    #structure = G.community_fastgreedy()
    #cl = structure.as_clustering().membership
    #Q = G.modularity(cl)
    Q = G.community_infomap().q
    return Q


def compute_coefficients(nl_fn, en_fn, bil_fn, out_fn, directed = False):

    nl = read(nl_fn, format="ncol")
    en = read(en_fn, format="ncol")
    bil = read(bil_fn, format="ncol")
    nl.simplify()
    en.simplify()
    bil.simplify()
    if not directed:
        nl.to_undirected()
        en.to_undirected()
        bil.to_undirected()
    nl_rand = Graph.Erdos_Renyi(n=len(nl.vs), m=len(nl.es), directed=directed, loops=False)
    en_rand = Graph.Erdos_Renyi(n=len(en.vs), m=len(en.es), directed=directed, loops=False)
    bil_rand = Graph.Erdos_Renyi(n=len(bil.vs), m=len(bil.es), directed=directed, loops=False)

    with open(out_fn, 'w') as f:
        f.write("N nl.:\t" + str(len(nl.vs)) + "\n")
        f.write("N en.:\t" + str(len(en.vs)) + "\n")
        f.write("N bil.:\t" + str(len(bil.vs)) + "\n")

    cc_nl = get_cc(nl, directed)
    cc_en = get_cc(en, directed)
    cc_bil = get_cc(bil, directed)

    with open(out_fn, 'a') as f:
        f.write("CC nl.:\t" + str(cc_nl) + "\n")
        f.write("CC en.:\t" + str(cc_en) + "\n")
        f.write("CC bil.:\t" + str(cc_bil) + "\n")

    aspl_nl = get_aspl(nl)
    aspl_en = get_aspl(en)
    aspl_bil = get_aspl(bil)

    with open(out_fn, 'a') as f:
        f.write("ASPL nl.:\t" + str(aspl_nl) + "\n")
        f.write("ASPL en.:\t" + str(aspl_en) + "\n")
        f.write("ASPL bil.:\t" + str(aspl_bil) + "\n")

    cc_nl_rand = get_cc(nl_rand, directed)
    cc_en_rand = get_cc(en_rand, directed)
    cc_bil_rand = get_cc(bil_rand, directed)

    with open(out_fn, 'a') as f:
        f.write("CC random nl.:\t" + str(cc_nl_rand) + "\n")
        f.write("CC random en.:\t" + str(cc_en_rand) + "\n")
        f.write("CC random bil.:\t" + str(cc_bil_rand) + "\n")

    aspl_nl_rand = get_aspl(nl_rand)
    aspl_en_rand = get_aspl(en_rand)
    aspl_bil_rand = get_aspl(bil_rand)
    with open(out_fn, 'a') as f:
        f.write("ASPL random nl.:\t" + str(aspl_nl_rand) + "\n")
        f.write("ASPL random en.:\t" + str(aspl_en_rand) + "\n")
        f.write("ASPL random bil.:\t" + str(aspl_bil_rand) + "\n")

    s_nl = (cc_nl / cc_nl_rand) / (aspl_nl / aspl_nl_rand)
    s_en = (cc_en / cc_en_rand) / (aspl_en / aspl_en_rand)
    s_bil = (cc_bil / cc_bil_rand) / (aspl_bil / aspl_bil_rand)

    with open(out_fn, 'a') as f:
        f.write("S nl.:\t" + str(s_nl) + "\n")
        f.write("S en.:\t" + str(s_en) + "\n")
        f.write("S bil.:\t" + str(s_bil) + "\n")

    Q_nl = get_q(nl)
    Q_en = get_q(en)
    Q_bil = get_q(bil)

    with open(out_fn, 'a') as f:
        f.write("Q nl.:\t" + str(Q_nl) + "\n")
        f.write("Q en.:\t" + str(Q_en) + "\n")
        f.write("Q bil.:\t" + str(Q_bil) + "\n")


compute_coefficients('./Dutch/shrunkdutch2.csv_dump',
                     './SF_norms/sothflorida_complete.csv_dump',
                     './biling_graph/orth_cogn_freq_True/biling_dump_L1_assoc_20_L2_assoc_10_TE_6_orth_5_asymm_1',
                     "./parameters.undirected",
                     directed = False)

compute_coefficients('./Dutch/shrunkdutch2.csv_dump',
                     './SF_norms/sothflorida_complete.csv_dump',
                     './biling_graph/orth_cogn_freq_True/biling_dump_L1_assoc_20_L2_assoc_10_TE_6_orth_5_asymm_1',
                     "./parameters.directed",
                     directed=True)

