import csv
import math
import ml_metrics as metrics
import numpy as np
from parameters import *
from scipy.stats import wilcoxon

def compute_differences(d1, d2, test_words):
    ups_max = []
    ups_n = []
    rhos_max = []
    rhos_n = []
    n_var = parameters["evaluation n value"]
    for w in sorted(test_words):
        n_max = min(len(d1[w]), len(d2[w]))
        tw1_sorted = sorted(d1[w].items(), key=lambda x: (-x[1], x[0]))
        tw2_sorted = sorted(d2[w].items(), key=lambda x: (-x[1], x[0]))
        lw1 = [e[0] for e in tw1_sorted]
        lw2 = [e[0] for e in tw2_sorted]
        dw1_max = normalize_dict(dict(tw1_sorted[:n_max]))
        dw2_max = normalize_dict(dict(tw2_sorted[:n_max]))
        dw1_n = normalize_dict(dict(tw1_sorted[:n_var]))
        dw2_n = normalize_dict(dict(tw2_sorted[:n_var]))
        upsilon_max = 0.5 * sum(abs((dw1_max.get(resp) or 0) - (dw2_max.get(resp) or 0)) for resp in sorted(set(dw1_max) | set(dw2_max)))
        upsilon_n = 0.5 * sum(abs((dw1_n.get(resp) or 0) - (dw2_n.get(resp) or 0)) for resp in sorted(set(dw1_n) | set(dw2_n)))

        sl, ll = sorted([(len(lw1), lw1), (len(lw2), lw2)])
        s, S = sl
        l, L = ll
        rho_max = 1 - metrics.apk(S, L, s)
        rho_n = 1 - metrics.apk(S[:n_var], L, n_var)

        # ups_max.append((w, upsilon_max))
        # ups_n.append((w, upsilon_n))
        # rhos_max.append((w, rho_max))
        # rhos_n.append((w, rho_n))
        ups_max.append(upsilon_max)
        ups_n.append(upsilon_n)
        rhos_max.append(rho_max)
        rhos_n.append(rho_n)
    return ups_max, ups_n, rhos_max, rhos_n


def print_difference_stats(ups_max_1, ups_n_1, rhos_max_1, rhos_n_1, ups_max_2, ups_n_2, rhos_max_2, rhos_n_2, f):

    f.write("RHO-3\n")
    #f.write("\tMEANS: within-group: %.3f, between-group, %.3f\n" % (np.mean([i[1] for i in rhos_n_1]), np.mean([i[1] for i in rhos_n_2])))
    f.write("\tMEANS: within-group: %.3f, between-group, %.3f\n" % (np.mean(rhos_n_1), np.mean(rhos_n_2)))
    #wilcoxon_result = wilcoxon([i[1] for i in rhos_n_1], [i[1] for i in rhos_n_2])
    wilcoxon_result = wilcoxon(rhos_n_1, rhos_n_2)
    f.write("\tWILCOXON SIGNED RANK TEST: T = %.3f, p = %.3f\n" % (wilcoxon_result[0], wilcoxon_result[1]))

    f.write("UPSILON-3\n")
    #f.write("\tMEANS: within-group: %.3f, between-group, %.3f\n" % (np.mean([i[1] for i in ups_n_1]), np.mean([i[1] for i in ups_n_2])))
    f.write("\tMEANS: within-group: %.3f, between-group, %.3f\n" % (np.mean(ups_n_1), np.mean(ups_n_2)))
    #wilcoxon_result = wilcoxon([i[1] for i in ups_n_1], [i[1] for i in ups_n_2])
    wilcoxon_result = wilcoxon(ups_n_1, ups_n_2)
    f.write("\tWILCOXON SIGNED RANK TEST: T = %.3f, p = %.3f\n" % (wilcoxon_result[0], wilcoxon_result[1]))

    f.write("RHO-MAX\n")
    #f.write("\tMEANS: within-group: %.3f, between-group, %.3f\n" % (np.mean([i[1] for i in rhos_max_1]), np.mean([i[1] for i in rhos_max_2])))
    f.write("\tMEANS: within-group: %.3f, between-group, %.3f\n" % (np.mean(rhos_max_1), np.mean(rhos_max_2)))
    #wilcoxon_result = wilcoxon([i[1] for i in rhos_max_1], [i[1] for i in rhos_max_2])
    wilcoxon_result = wilcoxon(rhos_max_1, rhos_max_2)
    f.write("\tWILCOXON SIGNED RANK TEST: T = %.3f, p = %.3f\n" % (wilcoxon_result[0], wilcoxon_result[1]))

    f.write("UPSILON-MAX\n")
    #f.write("\tMEANS: within-group: %.3f, between-group, %.3f\n" % (np.mean([i[1] for i in ups_max_1]), np.mean([i[1] for i in ups_max_2])))
    f.write("\tMEANS: within-group: %.3f, between-group, %.3f\n" % (np.mean(ups_max_1), np.mean(ups_max_2)))
    #wilcoxon_result = wilcoxon([i[1] for i in ups_max_1], [i[1] for i in ups_max_2])
    wilcoxon_result = wilcoxon(ups_max_1, ups_max_2)
    f.write("\tWILCOXON SIGNED RANK TEST: T = %.3f, p = %.3f\n" % (wilcoxon_result[0], wilcoxon_result[1]))


def read_alignments(fn):
    freq = {"en": {}, "nl": {}}
    with open(fn) as f:
        reader = csv.reader(f, skipinitialspace=True, quotechar=None, delimiter=",")
        next(reader)
        for row in reader:
            eng = row[0].strip().lower().strip("-")
            dut = row[1].strip().lower().strip("-")
            freq_ed = float(row[2].strip().strip("-"))
            freq_de = float(row[3].strip().strip("-"))
            if eng != "" and dut != "":
                eng = eng + ":en"
                dut = dut + ":nl"
                if freq_ed != 0:
                    freq["en"][(eng, dut)] = freq_ed
                if freq_de != 0:
                    freq["nl"][(dut, eng)] = freq_de
    return freq


def normalize_tuple_list(l, weight):
    # Given a list of tuples (word1, word2, weight), normalizes all weights, so that all edges outgoing from each word add up to the value of weight.
    if weight == 0:
        return []
    connections = {}
    for (w1, w2, c) in sorted(l):
        if w1 not in connections:
            connections[w1] = 0
        connections[w1] += c
    for idx in range(len(l)):
        w1 = l[idx][0]
        w2 = l[idx][1]
        c = l[idx][2]
        l[idx] = (w1, w2, weight*c/float(connections[w1]))
    return l


def normalize_tuple_dict(d, weight):
    # Given a dictionary {(word1, word2): weight}, normalizes all weights, so that all edges outgoing from each word add up to the value of weight.
    if weight == 0:
        return {}
    connections = {}
    for (w1, w2), c in sorted(d.items()):
        if w1 not in connections:
            connections[w1] = 0
        connections[w1] += c
    for (w1, w2), c in sorted(d.items()):
        d[(w1, w2)] = weight * c / float(connections[w1])
    return d


def normalize_dict(d, target=1.0):
    # Normalizes dictionary values to 1 (or another target value).
    raw = math.fsum(d.values())
    if raw == 0:
        return {}
    factor = target / raw
    for k in d:
        d[k] *= factor
    return d


def invert_dict(d):
    # Inverts a dictionary of {string:list} pairs.
    newdict = {}
    for k in d:
        for v in d[k]:
            newdict.setdefault(v, []).append(k)
    return newdict
