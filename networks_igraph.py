from collections import Counter
import copy
import csv
from igraph import *
import itertools
import ml_metrics as metrics
import numpy as np
import os
import utils
from pajek_converter import convert_pajek
import pandas
from parameters import extras, parameters
import random
from scipy.stats import ttest_rel
from test_data_reader import read_test_data
from numpy.random import choice

class LexNet:

    def __init__(self):
        self.G = Graph()
        self.alpha = parameters["spreading alpha"]
        self.min_freq = parameters["frequency threshold"]

    def clean_graph(self):
        # Removes noisy vertices from the graph.
        to_delete_ids = []
        for vx in self.G.vs:
            if vx["id"] in extras["noise list"]:
                to_delete_ids.append(vx.index)
            else:
                vx["name"] = vx["id"] + ":EN"
        self.G.delete_vertices(to_delete_ids)

    def spread_activation(self, responses, depth):
        # Main function that spreads the activation starting from the given node(s) and returns a dictionary of responses with their (non-normalized) likelihoods.
        if depth == 0:
            return ({})
        else:
            responses_current_level = dict()
            for vertex in sorted(responses):
                weight = responses[vertex]
                new = {}
                for e in self.G.incident(vertex):
                    if random.random() < self.alpha:
                        new[self.G.vs[self.G.es[e].tuple[1]]['name']] = self.G.es[e]["weight"]
                total = sum(new.values())
                if total == 0:
                    responses_single_word = {}
                else:
                    responses_single_word = {k: v / total * weight * self.alpha for k, v in new.items()}
                responses_current_level = dict(
                    sum((Counter(x) for x in [responses_current_level, responses_single_word]), Counter()))
            if not responses_current_level:
                return ({})
            else:
                responses_next_level = self.spread_activation(responses_current_level, depth - 1)
            final_vertices = dict(sum((Counter(x) for x in [responses_current_level, responses_next_level]), Counter()))
            return (final_vertices)

    def random_walk(self, responses, depth):
        if depth == 0:
            return (responses)
        else:
            new = dict()
            cue = list(responses.keys())[0]
            for e in self.G.incident(cue): 
                new[self.G.vs[self.G.es[e].tuple[1]]['name']] = self.G.es[e]["weight"]
            if not new:
                return {cue:1.0}
            draw = choice(list(new.keys()), 1, replace=False, p=list(new.values()))[0]
            return self.random_walk( {draw: 1.0} , depth-1 ) 

    def multiple_walks(self, responses, depth, iter_num=1000):
        return_stuff = {}
        for num in range(iter_num):
            draw = list(self.random_walk(responses, depth))[0]
            if draw in return_stuff:
                return_stuff[draw] = return_stuff[draw] + 1
            else:
                return_stuff[draw] = 1
        return_stuff = {k:v for k,v in return_stuff.items() if v > 1}
        return return_stuff

    def plot_activation(self, responses, vertices, edges, depth):
        # An auxiliary function that spreads activation for plotting a subgraph.
        if depth == 0:
            return ({}, vertices, edges)
        else:
            responses_current_level = dict()
            for vertex in sorted(responses):
                weight = responses[vertex]
                new = {}
                for e in self.G.incident(vertex):
                    if random.random() < self.alpha:
                        new[self.G.vs[self.G.es[e].tuple[1]]['name']] = self.G.es[e]["weight"]
                total = sum(new.values())
                if total == 0:
                    responses_single_word = {}
                else:
                    responses_single_word = {k: v / total * weight * self.alpha for k, v in new.items()}
                    for resp, wei in responses_single_word.items():
                        if resp not in vertices:
                            vertices.append(resp)
                            if (vertex, resp) not in edges:
                                edges[(vertex, resp)] = 0
                            edges[(vertex, resp)] += wei
                responses_current_level = dict(sum((Counter(x) for x in [responses_current_level, responses_single_word]), Counter()))
            if not responses_current_level:
                return ({}, vertices, edges)
            else:
                responses_next_level, vertices, edges = self.plot_activation(responses_current_level, vertices, edges, depth - 1)
            final_vertices = dict(sum((Counter(x) for x in [responses_current_level, responses_next_level]), Counter()))
            return (final_vertices, vertices, edges)

    def plot_subgraph(self, w, depth, name):
        # Traverses the graph for a given word and plots the resulting subgraph into a file. Needs to be tested.
        starting_vertex = {w: 1}
        #G_sub = Graph(directed=True)
        #G_sub.add_vertices(w)
        #G_sub.vs["name"] = [w]
        responses, vertices, edges = self.plot_activation(starting_vertex, [w], {}, depth)
        edges_filtered = [(k[0],k[1],v) for k,v in edges.items() if v > 0.001]
        G_sub = Graph.TupleList(edges=edges_filtered, edge_attrs="weight", directed=True)
        visual_style = {}
        visual_style["vertex_size"] = 10
        visual_style["vertex_label"] = G_sub.vs["name"]
        visual_style["edge_width"] = [x * 50 for x in G_sub.es['weight']]
        visual_style["bbox"] = (1000, 1000)
        visual_style["margin"] = 20
        plot(G_sub, "./plots/" + w + "_" + name + ".svg", **visual_style)

    def test_network(self, test_list, depth, direction, translation_dict=None, gold_full=None, verbose=True, log_file=None):
        # Tests the big network on a test list with words, using any language direction from Van Hell and De Groot (1998):
        # 'DD': Dutch stimulus, Dutch responses
        # 'DE': Dutch stimulus, English responses
        # 'ED': English stimulus, Dutch responses
        # 'EE': English stimulus, English responses
        # Returns three lists, one for each metrics:
        # - TVD (total variation distance)
        # - RBD (rank-biased distance)
        # - APK (average precision at k)
        # The length of each list equals to the length of the test word list.
        # Also prints responses if verbose is True.
        if translation_dict is None:
            translation_dict = {}
        gold = gold_full[direction]
        stimulus_lang = extras["language mapping"][direction[0]]
        response_lang = extras["language mapping"][direction[1]]
        tvds = []
        rbds = []
        apks = []
        apks_10 = []
        tvd = 0
        rbd = 0
        apk = 0
        apk_10 = 0
        for w in test_list:
            gold_local = dict(gold[w])
            gold_clean = {k: v for k, v in gold_local.items() if gold_local[k] > 1}
            d_gold = utils.normalize_dict(gold_clean)
            l_gold = sorted(d_gold.items(), key=lambda x: (-x[1], x[0]))
            k_gold = [pair[0] for pair in l_gold]
            starting_vertex = {w: 1.0}
            responses = self.spread_activation(starting_vertex, depth)
            #responses = self.multiple_walks(starting_vertex, depth)
            if w in responses:
                del responses[w]
            if stimulus_lang != response_lang:
                if w in translation_dict:
                    for trnsl in translation_dict[w]:
                        if trnsl in responses:
                            del responses[trnsl]
            responses_clean = {r: p for r, p in responses.items() if r[-3:] == response_lang}
            d_resp = utils.normalize_dict(responses_clean)
            l_resp = sorted(d_resp.items(), key=lambda x: (-x[1], x[0]))
            k_resp = [pair[0] for pair in l_resp]
            dm_resp = utils.normalize_dict({k: v for k, v in d_resp.items() if k in k_resp[:len(k_gold)]})
            if verbose:
                log_file.write("\tCUE: %s\n" % w)
                if gold:
                    log_file.write("\t\tGOLD\n")
                    for k, v in l_gold[:10]:
                        log_file.write("\t\t\t%s\t\t%.3f\n" % (k, v))
                log_file.write("\t\tMAX DEPTH: %d\n" % depth)
                for k, v in l_resp[:10]:
                    log_file.write("\t\t\t%s\t\t%.3f\n" % (k, v))
                log_file.flush()
            if gold:
                apk_k = len(gold_clean)
                tvd_w = 0.5 * sum(
                    abs((d_gold.get(resp) or 0) - (dm_resp.get(resp) or 0)) for resp in set(d_gold) | set(dm_resp))
                rbd_w = utils.get_rbd(k_gold, k_resp)
                apk_w = 1 - metrics.apk(k_gold, k_resp, apk_k)
                apk_w_10 = 1 - metrics.apk(k_gold, k_resp, 10)
                tvd += tvd_w
                rbd += rbd_w
                apk += apk_w
                apk_10 += apk_w_10
                if log_file is not None:
                    log_file.write("\t\tTVD: %.3f\n" % tvd_w)
                    log_file.write("\t\tRBD: %.3f\n" % rbd_w)
                    log_file.write("\t\tAPK(k): %.3f\n" % apk_w)
                    log_file.write("\t\tAPK(10): %.3f\n" % apk_w_10)
                tvds.append(tvd_w)
                rbds.append(rbd_w)
                apks.append(apk_w)
                apks_10.append(apk_w_10)
        if gold:
            tvd /= len(test_list)
            rbd /= len(test_list)
            apk /= len(test_list)
            apk_10 /= len(test_list)
            if log_file is not None:
                log_file.write("\tTotal variation distance: %.3f\n" % tvd)
                log_file.write("\tRank-biased distance: %.3f\n" % rbd)
                log_file.write("\tAverage precision (distance, k): %.3f\n" % apk)
                log_file.write("\tAverage precision (distance, 10): %.3f\n" % apk_10)
        if log_file is not None: log_file.flush()
        return (tvds, rbds, apks, apks_10)


class LexNetMo(LexNet):

    def __init__(self, fn, language):
        super().__init__()
        self.G = self.create_monolingual_graph(fn, language)
        self.lang = language

    def create_monolingual_graph(self, fn, language):
        if os.path.isfile(fn + "_dump"):
            #G = read(fn + "_dump", format="pickle")
            G = read(fn + "_dump", format="ncol")
        else:
            if language == "nl":
                G = self.construct_graph(fn, "NL")
            else: #language == "en"
                # if not os.path.isfile(fn + "_plain"):
                #     convert_pajek(fn)
                # G = self.construct_graph(fn + "_plain", "EN")
                G = self.construct_graph(fn, "EN")
            #G.write_pickle(fn + "_dump")
            G.write_ncol(fn + "_dump", names="name")
        return G

    def construct_graph(self, fn, lang):
        # Constructs the big monolingual graph from CSV.
        df = pandas.read_csv(fn, sep=";", na_values="", keep_default_na=False)
        df = df[(~df['cue'].str.contains(' ')) & (~df['asso1'].str.contains(' '))]
        edges = Counter()
        # for i in range(1, 4):
        # First response only:
        for i in range(1, 2):
            edges.update(zip(df['cue'], df['asso' + str(i)]))
        weighted_edges = [(e[0][0] + ":" + lang, e[0][1] + ":" + lang, e[1]) for e in edges.items() if
                          e[0][1] not in extras["noise list"] and e[1] > self.min_freq]
        weighted_edges = utils.normalize_tuple_list(weighted_edges, 1)
        G = Graph.TupleList(edges=weighted_edges, edge_attrs="weight", directed=True)
        return G


class LexNetBi(LexNet):

    def __init__(self, fn_l1, fn_l2, l2_l1_dic, L1_assoc_coeff, L2_assoc_coeff, TE_coeff, orth_coeff, asymm_ratio, mode):
        super().__init__()
        self.orth_threshold = parameters["orthographic threshold"]
        self.cogn_threshold = parameters["cognate threshold"]
        self.use_freq = parameters["use frequencies"]
        self.orth_edge_type = parameters["orth edge type"]
        self.G = self.construct_bilingual_graph(fn_l1, fn_l2, l2_l1_dic, L1_assoc_coeff, L2_assoc_coeff, TE_coeff, orth_coeff, asymm_ratio, mode)

    def get_assoc_edges(self, fn, lang, assoc_coeff):
        l = extras["language mapping"][lang]
        df = pandas.read_csv(fn, sep=";", na_values="", keep_default_na=False)
        df = df[(~df['cue'].str.contains(' ')) & (~df['asso1'].str.contains(' '))]
        edges = Counter()
        # for i in range(1, 4):
        # First response only:
        for i in range(1, 2):
            edges.update(zip(df['cue'], df['asso' + str(i)]))
        weighted_edges = [(e[0][0] + l, e[0][1] + l, e[1]) for e in edges.items() if e[0][1] not in
                             extras["noise list"] and e[1] > self.min_freq]
        vertices = [word + extras["language mapping"][lang] for word in
                    set.union(set(df['cue']), set(df['asso1']))]
                       #set.union(set(df['cue']), set(df['asso1']), set(df['asso2']), set(df['asso3']))]
        weighted_edges = utils.normalize_tuple_list(weighted_edges, assoc_coeff)
        return weighted_edges, vertices

    def get_orth_edges(self, vertices_en, vertices_nl, orth_coeff, mode, asymm_ratio):
        if mode == "orth":
            lev = utils.levLoader(self.orth_threshold, "./levdist.csv")
        elif mode == "cogn":
            lev = utils.levLoader(self.cogn_threshold, "./cognates.csv")
        else:
            sys.exit("MODE variable unknown. Unclear how to set orthographic links.")
        lev_edges_en_nl = {k: v for k, v in lev.items() if k[0] in vertices_en and k[1] in vertices_nl}
        lev_edges_nl_en = {(k[1], k[0]): v for k, v in lev_edges_en_nl.items()}
        lev_edges_en_nl = utils.normalize_tuple_dict(lev_edges_en_nl, orth_coeff * asymm_ratio)
        lev_edges_nl_en = utils.normalize_tuple_dict(lev_edges_nl_en, orth_coeff)
        lev_edges = copy.copy(lev_edges_en_nl)
        lev_edges.update(lev_edges_nl_en)

        return lev_edges

    def get_TE_edges(self, vertices_en, vertices_nl, en_nl_dic, TE_coeff, asymm_ratio):
        TE_all = [(en,nl) for en, nls in en_nl_dic.items() for nl in nls]
        if self.use_freq:
            freqs_nl = utils.read_frequencies("./frequencies/SUBTLEX-NL.txt", ":NL")
            freqs_en = utils.read_frequencies("./frequencies/en_google_ngrams", ":EN")
            TE_edges_en_nl = {tpl: (freqs_nl.get(tpl[1]) or 1) for tpl in TE_all if tpl[0] in vertices_en and tpl[1] in vertices_nl}
            TE_edges_nl_en = {(tpl[1], tpl[0]): (freqs_en.get(tpl[0]) or 1) for tpl in TE_edges_en_nl.keys()}
        else:
            TE_edges_en_nl = {tpl: 1 for tpl in TE_all if tpl[0] in vertices_en and tpl[1] in vertices_nl}
            TE_edges_nl_en = {(tpl[1], tpl[0]): 1 for tpl in TE_edges_en_nl.keys()}
        TE_edges_en_nl = utils.normalize_tuple_dict(TE_edges_en_nl, TE_coeff * asymm_ratio)
        TE_edges_nl_en = utils.normalize_tuple_dict(TE_edges_nl_en, TE_coeff)
        TE_edges = copy.copy(TE_edges_en_nl)
        TE_edges.update(TE_edges_nl_en)
        return TE_edges

    def construct_bilingual_graph(self, fn_nl, fn_en, en_nl_dic, L1_assoc_coeff, L2_assoc_coeff, TE_coeff, orth_coeff, asymm_ratio, mode):
        # Constructs a bilingual graph with various edges and pickles it. If a pickled file found, loads it instead.
        fn = "./biling_graph/orth_"+str(self.orth_edge_type)+"_freq_"+str(self.use_freq)+"/biling_dump_L1_assoc_" + str(L1_assoc_coeff) + "_L2_assoc_" + str(L2_assoc_coeff) + "_TE_" + str(TE_coeff) + "_orth_" + str(orth_coeff) + "_asymm_" + str(asymm_ratio)
        if os.path.isfile(fn):
            #biling = read(fn, format="pickle")
            biling = read(fn, format="ncol")
        else:

            edges_nl, vertices_nl = self.get_assoc_edges(fn_nl, "D", L1_assoc_coeff)
            # edges_en, vertices_en = self.get_assoc_edges(fn_en + "_plain", "E")
            edges_en, vertices_en = self.get_assoc_edges(fn_en, "E", L2_assoc_coeff)
            
            orth_edges = self.get_orth_edges(vertices_en, vertices_nl, orth_coeff, mode, asymm_ratio)
            TE_edges = self.get_TE_edges(vertices_en, vertices_nl, en_nl_dic, TE_coeff, asymm_ratio)
            crossling_edges = [(k[0], k[1], v) for k, v in (Counter(TE_edges) + Counter(orth_edges)).items()]

            edges = []
            edges.extend(edges_en)
            edges.extend(edges_nl)
            edges.extend(crossling_edges)
            edges = utils.normalize_tuple_list(edges, 1)

            biling = Graph.TupleList(edges=edges, edge_attrs="weight", directed=True)
            #biling.write_pickle(fn)
            biling.write_ncol(fn, names="name")
        return (biling)
