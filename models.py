import time
from collections import Counter
from igraph import *
from parameters import *
from utils import compute_differences, normalize_dict, normalize_tuple_dict, normalize_tuple_list

class LexNet:

    def __init__(self, model_type):
        self.G = Graph()
        self.lang = ""
        self.model_type = model_type
        self.min_freq = parameters["frequency threshold"]

    def create_edge_list(self, norms, language, kappa):
        edge_list = []
        for cue in norms:
            for resp in norms[cue]:
                prob = norms[cue][resp]
                edge_list.append((cue, resp, prob))
        edge_list = normalize_tuple_list(edge_list, kappa)
        return edge_list

    def spread_activation_ucs(self, resps, depth):
        # Main function that spreads the activation starting from the given node(s) and returns a dictionary of resps with their (non-normalized) likelihoods.
        if depth == 0:
            return {}
        else:
            resps_current_level = dict()
            for vertex in sorted(resps):
                weight = resps[vertex]
                new = {}
                for e in self.G.incident(vertex):
                    adjacent_vertex = self.G.vs[self.G.es[e].tuple[1]]['name']
                    new[adjacent_vertex] = self.G.es[e]["weight"]
                total = sum(new.values())
                if total == 0:
                    resps_single_word = {}
                else:
                    resps_single_word = {k: v / total * weight for k, v in new.items()}
                resps_current_level = dict(sum((Counter(x) for x in [resps_current_level, resps_single_word]), Counter()))
            if not resps_current_level:
                return {}
            else:
                resps_next_level = self.spread_activation_ucs(resps_current_level, depth - 1)
            final_vertices = dict(sum((Counter(x) for x in [resps_current_level, resps_next_level]), Counter()))
            return final_vertices

    def spread_activation_cs(self, resps, depth, path_length):
        # Main function that spreads the activation starting from the given node(s) and returns a dictionary of resps with their (non-normalized) likelihoods.
        if depth == 0:
            return {}
        else:
            resps_current_level = dict()
            for vertex in sorted(resps):
                weight = resps[vertex]
                new = {}
                for e in self.G.incident(vertex):
                    adjacent_vertex = self.G.vs[self.G.es[e].tuple[1]]['name']
                    new[adjacent_vertex] = self.G.es[e]["weight"]
                total = sum(new.values())
                if total == 0:
                    resps_single_word = {}
                else:
                    resps_single_word = {k: v / total * weight for k, v in new.items()}
                resps_current_level = dict(sum((Counter(x) for x in [resps_current_level, resps_single_word]), Counter()))
            if not resps_current_level:
                return {}
            else:
                if (path_length - depth == 0):
                    resps_next_level = self.spread_activation_cs({k:v for k,v in resps_current_level.items() if k[-3:] == ":nl"}, depth - 1, path_length)
                else:
                    if (path_length - depth == 1):
                        resps_current_level = {k:v for k,v in resps_current_level.items() if k[-3:] == ":nl"}
                    elif (path_length - depth == 2):
                        resps_current_level = {k:v for k,v in resps_current_level.items() if k[-3:] == ":en"}
                    resps_next_level = self.spread_activation_cs(resps_current_level, depth - 1, path_length)
            final_vertices = dict(sum((Counter(x) for x in [resps_current_level, resps_next_level]), Counter()))
            return final_vertices


    def plot_activation(self, resps, vertices, edges, depth):
        # An auxiliary function that spreads activation for plotting a subgraph.
        if depth == 0:
            return ({}, vertices, edges)
        else:
            resps_current_level = dict()
            for vertex in sorted(resps):
                weight = resps[vertex]
                new = {}
                for e in self.G.incident(vertex):
                    adjacent_vertex = self.G.vs[self.G.es[e].tuple[1]]['name']
                    new[adjacent_vertex] = self.G.es[e]["weight"]
                total = sum(new.values())
                if total == 0:
                    resps_single_word = {}
                else:
                    resps_single_word = {k: v / total * weight for k, v in new.items()}
                    for resp, wei in resps_single_word.items():
                        if resp not in vertices:
                            vertices.append(resp)
                            if (vertex, resp) not in edges:
                                edges[(vertex, resp)] = 0
                            edges[(vertex, resp)] += wei
                resps_current_level = dict(sum((Counter(x) for x in [resps_current_level, resps_single_word]), Counter()))
            if not resps_current_level:
                return ({}, vertices, edges)
            else:
                resps_next_level, vertices, edges = self.plot_activation(resps_current_level, vertices, edges, depth - 1)
            final_vertices = dict(sum((Counter(x) for x in [resps_current_level, resps_next_level]), Counter()))
            return (final_vertices, vertices, edges)

    def plot_subgraph(self, w, depth, name):
        # Traverses the graph for a given word and plots the resulting subgraph into a file. Needs to be tested.
        starting_vertex = {w: 1}
        resps, vertices, edges = self.plot_activation(starting_vertex, [w], {}, depth)
        edges_filtered = [(k[0],k[1],v) for k,v in edges.items() if v > 0.01]
        G_sub = Graph.TupleList(edges=edges_filtered, edge_attrs="weight", directed=True)
        visual_style = {}
        visual_style["vertex_size"] = 10
        visual_style["vertex_label"] = G_sub.vs["name"]
        visual_style["edge_width"] = [x * 50 for x in G_sub.es['weight']]
        visual_style["bbox"] = (1000, 1000)
        visual_style["margin"] = 20
        plot(G_sub, "./plots/" + w + "_" + name + ".svg", **visual_style)

    def test_network(self, test_cues, depth, resp_lang, resps_gold=None, verbose=True, log_file=None):
        resps_predicted = {}
        for cue in test_cues:
            if cue not in self.G.vs['name']:
                resps_clean = {}
            else:
                starting_vertex = {cue: 1.0}

                if self.lang == "en":
                    resps = self.spread_activation_ucs(starting_vertex, depth)
                elif self.lang == "nl-en":
                    if self.model_type == "cs":
                        resps = self.spread_activation_cs(starting_vertex, parameters["spreading depth"], parameters["spreading depth"])
                    else:
                        resps = self.spread_activation_ucs(starting_vertex, parameters["spreading depth"])
                else:
                    sys.exit("Unknown network language.")

                if None in resps:
                    del resps[None]
                if cue in resps:
                    del resps[cue]
                resps_clean = normalize_dict({r: p for r, p in resps.items() if r[-2:] == resp_lang})

            resps_predicted[cue] = resps_clean
            if verbose:
                log_file.write("\tCUE: %s\n" % cue)
                if resps_gold:
                    log_file.write("\t\tGOLD\n")
                    gold_top = sorted(resps_gold[cue].items(), key=lambda x: (-x[1], x[0]))
                    for k, v in gold_top[:10]:
                        log_file.write("\t\t\t%s\t\t%.3f\n" % (k, v))
                pred_top = sorted(resps_predicted[cue].items(), key=lambda x: (-x[1], x[0]))
                log_file.write("\t\tPREDICTED\n")
                for k, v in pred_top[:10]:
                    log_file.write("\t\t\t%s\t\t%.3f\n" % (k, v))
                log_file.flush()

        ups_max, ups_n, rhos_max, rhos_n = compute_differences(resps_gold, resps_predicted, test_cues)

        return ups_max, ups_n, rhos_max, rhos_n


class LexNetMo(LexNet):

    def __init__(self, model_type, norms, language):
        super().__init__(model_type)
        self.G = self.create_monolingual_graph(norms, language)
        self.lang = language

    def create_monolingual_graph(self, norms, language):
        model_fn = "%s/model_mo_%s" % (parameters["edge directory"], language)
        if os.path.isfile(model_fn):
            G = read(model_fn, format="ncol")
        else:
            G = self.construct_graph(norms, language)
            G.write_ncol(model_fn, names="name")
        return G

    def construct_graph(self, norms, language):
        edge_list = self.create_edge_list(norms, language, 1.0)
        G = Graph.TupleList(edges=edge_list, edge_attrs="weight", directed=True)
        return G


class LexNetBi(LexNet):

    def __init__(self, vertices_nl, vertices_en, norms_nl, norms_en, orth_sims_en, synt_coocs_en, en_nl_dic, cognates, alignments, k_da, k_ea, k_te, k_cg, k_or, k_sy, model_type):
        super().__init__(model_type)
        self.lang = "nl-en"
        self.orth_threshold = parameters["orthographic threshold"]
        self.cogn_threshold = parameters["cognate threshold"]
        self.G = self.construct_bilingual_graph(vertices_nl, vertices_en, norms_nl, norms_en, orth_sims_en, synt_coocs_en, cognates, alignments, k_da, k_ea, k_te, k_cg, k_or, k_sy)

    def get_assoc_edges(self, norms, language, k_a, convert_to_dict):
        if k_a == 0:
            return {}
        edges_a_list = self.create_edge_list(norms, language, k_a)
        if convert_to_dict:
            edges_a = dict(((a, b), c) for a, b, c in edges_a_list)
            return edges_a
        return edges_a_list

    def get_other_edges(self, tuple_dict, kappa):
        if kappa == 0:
            return {}
        edges = normalize_tuple_dict(tuple_dict, kappa)
        return edges

    def construct_bilingual_graph(self, vertices_nl, vertices_en, norms_nl, norms_en, orth_sims_en, synt_coocs_en, cognates, alignments, k_da, k_ea, k_te, k_cg, k_or, k_sy):

        model_fn = "%s/model_da_%s_ea_%s_te_%s_cg_%s_or_%s_sy_%s" % (parameters["edge directory"], str(k_da), str(k_ea), str(k_te), str(k_cg), str(k_or), str(k_sy))

        delta_time = 0
        if os.path.isfile(model_fn):
            delta_time = time.time() - os.path.getmtime(model_fn)
        if delta_time > 300:
            biling = read(model_fn, format="ncol")
        else:

            da_edges = self.get_assoc_edges(norms_nl, "nl", k_da, False)
            edges_all_nl = da_edges

            ea_edges = self.get_assoc_edges(norms_en, "en", k_ea, True)
            or_edges = self.get_other_edges(orth_sims_en, k_or)
            sy_edges = self.get_other_edges(synt_coocs_en, k_sy)
            edges_all_en = [(k[0], k[1], v) for k, v in (Counter(ea_edges) + Counter(or_edges) + Counter(sy_edges)).items()]

            cg_edges = self.get_other_edges(cognates, k_cg)
            te_edges = self.get_other_edges(alignments, k_te)
            edges_all_xling = [(k[0], k[1], v) for k, v in (Counter(te_edges) + Counter(cg_edges)).items()]

            edges_all = []
            edges_all.extend(edges_all_nl)
            edges_all.extend(edges_all_en)
            edges_all.extend(edges_all_xling)
            edges_all = normalize_tuple_list(edges_all, 1.0)

            biling = Graph.TupleList(edges=edges_all, edge_attrs="weight", directed=True)
            biling.add_vertices(vertices_en)
            biling.add_vertices(vertices_nl)
            biling.write_ncol(model_fn, names="name")

        return biling