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
        self.alpha = parameters["activation decay"]
        self.min_freq = parameters["frequency threshold"]
        self.num_walks = parameters["number of walks"]
        self.retrieval_algorighm = parameters["retrieval algorithm"]
        self.return_allowed = parameters["return allowed"]
        self.syntagmatic_edges = parameters["use syntagmatic edges"]

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
            return {}
        else:
            responses_current_level = dict()
            for vertex in sorted(responses):
                weight = responses[vertex]
                new = {}
                for e in self.G.incident(vertex):
                    adjacent_vertex = self.G.vs[self.G.es[e].tuple[1]]['name']
                    if self.return_allowed or adjacent_vertex not in responses:
                        new[adjacent_vertex] = self.G.es[e]["weight"]
                total = sum(new.values())
                if total == 0:
                    responses_single_word = {}
                else:
                    responses_single_word = {k: v / total * weight for k, v in new.items()}
                responses_current_level = dict(sum((Counter(x) for x in [responses_current_level, responses_single_word]), Counter()))
            if not responses_current_level:
                return {}
            else:
                responses_next_level = self.spread_activation(responses_current_level, depth - 1)
            final_vertices = dict(sum((Counter(x) for x in [responses_current_level, responses_next_level]), Counter()))
            return final_vertices

    def random_walk(self, responses, depth):
        if depth == 0:
            return (responses)
        else:
            new = dict()
            cue = list(responses.keys())[0]
            for e in self.G.incident(cue):
                adjacent_vertex = self.G.vs[self.G.es[e].tuple[1]]['name']
                if (random.random() < self.alpha) and (self.return_allowed or adjacent_vertex not in responses):
                    new[adjacent_vertex] = self.G.es[e]["weight"]
            if not new:
                return {cue:1.0}
            scores = np.array(list(new.values()))
            draw = choice(list(new.keys()), 1, replace=False, p=scores/np.sum(scores))[0]
            return self.random_walk( {draw: 1.0} , depth-1 ) 

    def random_walk_with_stopwords(self, cue, depth, alpha, stopwords, response_lang):
        if depth == 0 or random.random() >= alpha:
            response = cue
            if response is None or ((response not in stopwords) and (response[-3:] == response_lang)):
                return response
            else:
                return self.random_walk_with_stopwords( response , 1, 1, stopwords, response_lang)
        else:
            new = dict()
            for e in self.G.incident(cue):
                adjacent_vertex = self.G.vs[self.G.es[e].tuple[1]]['name']
                if self.return_allowed or adjacent_vertex != cue:
                    new[adjacent_vertex] = self.G.es[e]["weight"]
            if not new:
                return None
            scores = np.array(list(new.values()))
            response = choice(list(new.keys()), 1, replace=False, p=scores/np.sum(scores))[0]
            return self.random_walk_with_stopwords( response , depth-1, alpha, stopwords, response_lang )

    def multiple_walks(self, responses, depth):
        final_sample = {}
        for num in range(self.num_walks):
            draw = list(self.random_walk(responses, depth))[0]
            if draw in final_sample:
                final_sample[draw] += 1
            else:
                final_sample[draw] = 1
        final_sample_filtered = {k:v for k,v in final_sample.items() if v > 1}
        return final_sample_filtered

    def multiple_walks_with_stopwords(self, cue, depth, alpha, stopwords, response_lang):
        final_sample = {}
        for num in range(self.num_walks):
            draw = self.random_walk_with_stopwords(cue, depth, alpha, stopwords, response_lang)
            if draw is not None:
                if draw in final_sample:
                    final_sample[draw] += 1
                else:
                    final_sample[draw] = 1
        final_sample_filtered = {k:v for k,v in final_sample.items() if v > 1}
        return final_sample_filtered

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
                    adjacent_vertex = self.G.vs[self.G.es[e].tuple[1]]['name']
                    if self.return_allowed or adjacent_vertex not in responses:
                        new[adjacent_vertex] = self.G.es[e]["weight"]
                total = sum(new.values())
                if total == 0:
                    responses_single_word = {}
                else:
                    responses_single_word = {k: v / total * weight for k, v in new.items()}
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
        edges_filtered = [(k[0],k[1],v) for k,v in edges.items() if v > 0.005]
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
        jacs = []
        apks = []
        apks_10 = []
        tvd = 0
        rbd = 0
        jac = 0
        apk = 0
        apk_10 = 0
        for w in test_list:
            gold_local = dict(gold[w])
            gold_clean = utils.normalize_dict({k: v for k, v in gold_local.items() if gold_local[k] > self.min_freq})

            t_gold_sorted = sorted(gold_clean.items(), key=lambda x: (-x[1], x[0]))
            l_gold = [e[0] for e in t_gold_sorted]
            s_gold = set(l_gold[:parameters["jaccard k"]])

            if w not in self.G.vs['name']:
                responses_clean = {}
            else:
                starting_vertex = {w: 1.0}
                if self.retrieval_algorighm == "spreading":
                    responses = self.spread_activation(starting_vertex, depth)
                elif self.retrieval_algorighm == "walks":
                    #responses = self.multiple_walks(starting_vertex, depth)
                    responses = self.multiple_walks_with_stopwords(w, depth, self.alpha, [w], response_lang)
                else:
                    sys.exit("Unknown retrieval algorithm.")
                if None in responses:
                    del responses[None]
                if w in responses:
                    del responses[w]
                if stimulus_lang != response_lang:
                    if w in translation_dict:
                        for trnsl in translation_dict[w]:
                            if trnsl in responses:
                                del responses[trnsl]
                responses_clean = utils.normalize_dict({r: p for r, p in responses.items() if r[-3:] == response_lang})

            t_resp_sorted = sorted(responses_clean.items(), key=lambda x: (-x[1], x[0]))
            l_resp = [e[0] for e in t_resp_sorted]
            s_resp = set(l_resp[:parameters["jaccard k"]])

            k = min(len(gold_clean), len(responses_clean))
            d_gold = utils.normalize_dict(dict(t_gold_sorted[:k]))
            d_resp = utils.normalize_dict(dict(t_resp_sorted[:k]))

            if verbose:
                log_file.write("\tCUE: %s\n" % w)
                if gold:
                    log_file.write("\t\tGOLD\n")
                    for k, v in t_gold_sorted[:15]:
                        log_file.write("\t\t\t%s\t\t%.3f\n" % (k, v))
                log_file.write("\t\tMAX DEPTH: %d\n" % depth)
                for k, v in t_resp_sorted[:15]:
                    log_file.write("\t\t\t%s\t\t%.3f\n" % (k, v))
                log_file.flush()
            if gold:
                if len(d_resp) == 0:
                    tvd_w = 1
                else:
                    tvd_w = 0.5 * sum(abs((d_gold.get(resp) or 0) - (d_resp.get(resp) or 0)) for resp in set(d_gold) | set(d_resp))
                rbd_w = utils.get_rbd(l_gold, l_resp)
                jac_w = 1 - len(set.intersection(s_gold,s_resp))/float(len(set.union(s_gold, s_resp)))
                apk_w = 1 - metrics.apk(l_gold, l_resp, len(l_gold))
                apk_w_10 = 1 - metrics.apk(l_gold[:parameters["apk k"]], l_resp, parameters["apk k"])
                tvd += tvd_w
                rbd += rbd_w
                jac += jac_w
                apk += apk_w
                apk_10 += apk_w_10
                if log_file is not None:
                    log_file.write("\t\tTVD: %.3f\n" % tvd_w)
                    log_file.write("\t\tRBD: %.3f\n" % rbd_w)
                    log_file.write("\t\tJAC: %.3f\n" % jac_w)
                    log_file.write("\t\tAPK(k): %.3f\n" % apk_w)
                    log_file.write("\t\tAPK(10): %.3f\n" % apk_w_10)
                tvds.append(tvd_w)
                rbds.append(rbd_w)
                jacs.append(jac_w)
                apks.append(apk_w)
                apks_10.append(apk_w_10)
        if gold:
            tvd /= len(test_list)
            rbd /= len(test_list)
            jac /= len(test_list)
            apk /= len(test_list)
            apk_10 /= len(test_list)
            if log_file is not None:
                log_file.write("\tTotal variation distance: %.3f\n" % tvd)
                log_file.write("\tRank-biased distance: %.3f\n" % rbd)
                log_file.write("\tJaccard distance: %.3f\n" % jac)
                log_file.write("\tAverage precision (distance, k): %.3f\n" % apk)
                log_file.write("\tAverage precision (distance, 10): %.3f\n" % apk_10)
        if log_file is not None: log_file.flush()
        return (tvds, rbds, jacs, apks, apks_10)


    def test_network_single_walk(self, test_list, depth, direction, translation_dict=None):
        if translation_dict is None:
            translation_dict = {}
        response_lang = extras["language mapping"][direction[1]]
        responses = {}
        for w in test_list:
            if w in self.G.vs['name']:
                starting_vertex = {w: 1.0}
                stopwords = [w] + (translation_dict.get(w) or [])
                response = self.random_walk_with_stopwords(w, depth, self.alpha, stopwords, response_lang)
                # response = list(self.random_walk_with_stopwords(starting_vertex, depth, self.alpha, stopwords, response_lang).keys())[0]
                if response is not None:
                    responses[w] = response
        return responses


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

    def __init__(self, fn_l1, fn_l2, l2_l1_dic, L1_assoc_coeff, L2_assoc_coeff, TE_coeff, orth_coeff, synt_coeff, asymm_ratio, mode):
        super().__init__()
        self.orth_threshold = parameters["orthographic threshold"]
        self.cogn_threshold = parameters["cognate threshold"]
        self.use_freq = parameters["use frequencies"]
        self.orth_edge_type = parameters["orth edge type"]
        self.freq_mode = parameters["frequency mode"]
        self.G = self.construct_bilingual_graph(fn_l1, fn_l2, l2_l1_dic, L1_assoc_coeff, L2_assoc_coeff, TE_coeff, orth_coeff, synt_coeff, asymm_ratio, mode)

    def get_assoc_edges(self, fn, lang, assoc_coeff):
        l = extras["language mapping"][lang]
        df = pandas.read_csv(fn, sep=";", na_values="", keep_default_na=False)
        df = df[(~df['cue'].str.contains(' ')) & (~df['asso1'].str.contains(' '))]
        edges = Counter()
        # for i in range(1, 4):
        # First response only:
        for i in range(1, 2):
            edges.update(zip(df['cue'], df['asso' + str(i)]))
        weighted_edges = {(e[0][0] + l, e[0][1] + l): e[1] for e in edges.items() if e[0][1] not in
                             extras["noise list"] and e[1] > self.min_freq}
        # vertices = [word + extras["language mapping"][lang] for word in
        #             set.union(set(df['cue']), set(df['asso1']))]
        vertices = [word for word in
                    set.union(set([i[0] for i in weighted_edges.keys()]), (set([i[1] for i in weighted_edges.keys()])))]
        #set.union(set(df['cue']), set(df['asso1']), set(df['asso2']), set(df['asso3']))]
        weighted_edges = utils.normalize_tuple_dict(weighted_edges, assoc_coeff)
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

    def get_synt_edges(self, vertices_en, synt_coeff):
        synt_weights = utils.syntLoader("./coca/ngrams.lemmas.cond.prob.filtered.csv", vertices_en)
        #synt_weights_inverse = {(tpl[1], tpl[0]): w for tpl,w in synt_weights.items()}
        #synt_weights.update(synt_weights_inverse)
        synt_edges = utils.normalize_tuple_dict(synt_weights, synt_coeff)

        return synt_edges

    def get_TE_edges(self, vertices_en, vertices_nl, en_nl_dic, TE_coeff, asymm_ratio):
        TE_all = [(en,nl) for en, nls in en_nl_dic.items() for nl in nls]
        if self.use_freq:
            if self.freq_mode == "unigrams":
                freqs_nl = utils.read_frequencies("./frequencies/SUBTLEX-NL.txt", ":NL")
                freqs_en = utils.read_frequencies("./frequencies/en_google_ngrams", ":EN")
                TE_edges_en_nl = {tpl: (freqs_nl.get(tpl[1]) or 1) for tpl in TE_all if tpl[0] in vertices_en and tpl[1] in vertices_nl}
                TE_edges_nl_en = {(tpl[1], tpl[0]): (freqs_en.get(tpl[0]) or 1) for tpl in TE_edges_en_nl.keys()}
            else:
                freqs = utils.read_alignment_frequencies("./alignment/en-nl.refined.dic.lemmas.csv")
                TE_edges_en_nl = {tpl: (freqs.get(tpl) or 1) for tpl in TE_all if
                                  tpl[0] in vertices_en and tpl[1] in vertices_nl}
                TE_edges_nl_en = {(tpl[1], tpl[0]): w for tpl,w in TE_edges_en_nl.items()}
        else:
            TE_edges_en_nl = {tpl: 1 for tpl in TE_all if tpl[0] in vertices_en and tpl[1] in vertices_nl}
            TE_edges_nl_en = {(tpl[1], tpl[0]): 1 for tpl in TE_edges_en_nl.keys()}
        TE_edges_en_nl = utils.normalize_tuple_dict(TE_edges_en_nl, TE_coeff * asymm_ratio)
        TE_edges_nl_en = utils.normalize_tuple_dict(TE_edges_nl_en, TE_coeff)
        TE_edges = copy.copy(TE_edges_en_nl)
        TE_edges.update(TE_edges_nl_en)
        return TE_edges

    def construct_bilingual_graph(self, fn_nl, fn_en, en_nl_dic, L1_assoc_coeff, L2_assoc_coeff, TE_coeff, orth_coeff, synt_coeff, asymm_ratio, mode):
        # Constructs a bilingual graph with various edges and pickles it. If a pickled file found, loads it instead.
        fn = parameters["edge directory"]+"orth_" + str(self.orth_edge_type)+\
             "_freq_" + str(self.use_freq) + \
             "/biling_dump_L1_assoc_" + str(L1_assoc_coeff) + \
             "_L2_assoc_" + str(L2_assoc_coeff) + \
             "_TE_" + str(TE_coeff) + \
             "_orth_" + str(orth_coeff) + \
             "_synt_" + str(synt_coeff) + \
             "_asymm_" + str(asymm_ratio)
        if os.path.isfile(fn):
            #biling = read(fn, format="pickle")
            biling = read(fn, format="ncol")
        else:

            edges_assoc_nl, vertices_nl = self.get_assoc_edges(fn_nl, "D", L1_assoc_coeff)
            edges_nl = [(k[0],k[1],v) for k,v in edges_assoc_nl.items()]

            edges_assoc_en, vertices_en = self.get_assoc_edges(fn_en, "E", L2_assoc_coeff)
            edges_synt_en = self.get_synt_edges(vertices_en, synt_coeff)
            edges_en = [(k[0], k[1], v) for k, v in (Counter(edges_assoc_en) + Counter(edges_synt_en)).items()]

            orth_edges = self.get_orth_edges(vertices_en, vertices_nl, orth_coeff, mode, asymm_ratio)
            TE_edges = self.get_TE_edges(vertices_en, vertices_nl, en_nl_dic, TE_coeff, asymm_ratio)
            crossling_edges = [(k[0], k[1], v) for k, v in (Counter(TE_edges) + Counter(orth_edges)).items()]

            edges = []
            edges.extend(edges_nl)
            edges.extend(edges_en)
            edges.extend(crossling_edges)
            edges = utils.normalize_tuple_list(edges, 1)

            biling = Graph.TupleList(edges=edges, edge_attrs="weight", directed=True)
            biling.add_vertices(vertices_en)
            biling.add_vertices(vertices_nl)
            #biling.write_pickle(fn)
            biling.write_ncol(fn, names="name")
        return (biling)
