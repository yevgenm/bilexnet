mapping = {'E': ":EN", 'D': ":NL"}
from collections import Counter
from scipy.stats import ttest_rel
from utils import get_rbd
import csv
import numpy as np
import ml_metrics as metrics
import pandas
import pickle
from scipy.stats import wilcoxon

n_abs_ncog = ["duty","opportunity","favour","attempt","advantage","ease","revenge","truth","conscience","cause","memory","demand","care","faith","property"]
n_abs_cog = ["insight","shame","plan","motive","quality","hell","block","method","chance","principle","information","metal","circle","panic","figure"]
n_con_ncog = ["shop","mirror","gun","potato","knife","bottle","skirt","flower","tree","hospital","trousers","farm","bird","bike","jail"]
n_con_cog = ["shoulder","season","finger","captain","daughter","pepper","slave","apple","snow","winter","coffee","rose","police","train","doctor"]
v_abs_ncog = ["refuse","admit","disturb","succeed","guess","promise","understand"]
v_abs_cog = ["forget","bend","dare","arrest","hate","hope","spread","irritate"]
v_con_ncog = ["tremble","move","baptize","throw","calculate","marry","cry","paint"]
v_con_cog = ["sneeze","climb","frown","swim","listen","steal","sing"]
nouns = n_abs_ncog + n_abs_cog + n_con_ncog + n_con_cog
verbs = v_abs_ncog + v_abs_cog + v_con_ncog + v_con_cog
abst = n_abs_ncog + n_abs_cog + v_abs_ncog + v_abs_cog
con = n_con_ncog + n_con_cog + v_con_ncog + v_con_cog
ncog = n_abs_ncog + n_con_ncog + v_abs_ncog + v_con_ncog
cog = n_abs_cog + n_con_cog + v_abs_cog + v_con_cog

np.random.seed(12345678)
threshold_frequency = 0
noise_list = ["x", "", None]

def read_norm_data(fn):
    # Constructs the big graph from CSV.
    df = pandas.read_csv(fn, sep=";", na_values="", keep_default_na=False)
    tuples = Counter()
    for i in range(1,4):
        tuples.update(zip(df['cue'], df['asso'+str(i)]))
    dict = {}
    for t,v in tuples.items():
        if v > threshold_frequency and t[1] not in noise_list:
            if t[0] not in dict:
                dict[t[0]] = {}
            if t[1] not in dict[t[0]]:
                dict[t[0]][t[1]] = 0
            dict[t[0]][t[1]] += v
    return dict

def normalize_dict(d, target=1.0):
    # Normalizes dictionary values to 1.
    raw = sum(d.values())
    if raw == 0: return {}
    factor = target / raw
    return {key: value * factor for key, value in d.items()}

def preprocess_word(w):
    # An auxiliary function to clean test files.
    if w[0:2] == "vk":
        w = w[2:]
    if w=="geen":
        return None
    if "(" in w:
        w = w[:w.index("(")]
    return(w)

def read_test_file(fn):
    # An auxiliary function that reads a single test file.
    conditions = fn.split("/")[-1].split('.')[0].split('-')
    cue_langs = [c[0] for c in conditions]
    cue_langs = [cue_langs[idx]+str(idx) for idx in range(len(cue_langs))]
    target_langs = [c[1] for c in conditions]
    target_langs = [target_langs[idx] + str(idx) for idx in range(len(target_langs))]
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

def read_test_data(condition):
    # Reads test data of Van Hell & De Groot (1998). CSV format needed.

    if condition[1] == "D":
        DD_DE_DD_test_dict = read_test_file("./vanhell/DD1-DE2-DD3.csv")
        DE_DD_test_dict = read_test_file("./vanhell/DE1-DD2.csv")

        test_lists = [DD_DE_DD_test_dict['D0']['D0'],
                          DD_DE_DD_test_dict['D2']['D2'],
                          DE_DD_test_dict['D1']['D1']]

    elif condition[1] == "E":
        EE_ED_EE_test_dict = read_test_file("./vanhell/EE1-ED2-EE3.csv")
        ED_EE_test_dict = read_test_file("./vanhell/ED1-EE2.csv")

        test_lists = [EE_ED_EE_test_dict['E0']['E0'],
                      EE_ED_EE_test_dict['E2']['E2'],
                      ED_EE_test_dict['E1']['E1']]

    gold_dict = {}
    for session in range(len(test_lists)):
        for k,v in Counter(test_lists[session]).items():
            if session not in gold_dict:
                gold_dict[session] = {}
            if k[0] not in gold_dict[session]:
                gold_dict[session][k[0]] = {}
            if v > threshold_frequency:
                gold_dict[session][k[0]][k[1]] = v
        for d in gold_dict[session]:
            gold_dict[session][d] = normalize_dict(gold_dict[session][d])

    return(gold_dict)


def get_diff(d1, d2, test_words):
    tvds = []
    rbds = []
    apks = []
    apks10 = []
    for w in sorted(test_words):
        k = min(len(d1[w]), len(d2[w]))
        dw1 = sorted(d1[w].items(), key=lambda x: (-x[1], x[0]))
        dw2 = sorted(d2[w].items(), key=lambda x: (-x[1], x[0]))
        lw1 = [e[0] for e in dw1]
        lw2 = [e[0] for e in dw2]
        dw1 = normalize_dict(dict(dw1[:k]))
        dw2 = normalize_dict(dict(dw2[:k]))
        tvd = 0.5 * sum(abs((dw1.get(resp) or 0) - (dw2.get(resp) or 0)) for resp in set(dw1) | set(dw2))
        rbd = get_rbd(lw1, lw2)
        apk = 1-metrics.apk(lw1, lw2, k)
        apk10 = 1 - metrics.apk(lw1, lw2, 10)
        tvds.append((w, tvd))
        rbds.append((w, rbd))
        apks.append((w, apk))
        apks10.append((w, apk10))
    return tvds, rbds, apks, apks10

def read_aggregated_test_file(fn):
    # An auxiliary function that reads a single test file.
    conditions = fn.split("/")[-1].split('.')[0].split('-')
    cue_langs = [c[0] for c in conditions]
    cue_langs = [cue_langs[idx] for idx in range(len(cue_langs))]
    target_langs = [c[1] for c in conditions]
    target_langs = [target_langs[idx] for idx in range(len(target_langs))]
    n_conds = len(conditions)
    resp_dict = {}
    with open(fn) as f:
        test_reader = csv.reader(f, delimiter=",")
        next(test_reader)
        for row in test_reader:
            cue = row[1]
            responses_mixed = row[3:]
            #for cond_idx in range(n_conds):
            for cond_idx in range(2):
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

def read_aggregated_test_data():
    # Reads test data of Van Hell & De Groot (1998). CSV format needed.

    DD_DE_DD_test_dict = read_aggregated_test_file("./vanhell/DD1-DE2-DD3.csv")
    DE_DD_test_dict = read_aggregated_test_file("./vanhell/DE1-DD2.csv")
    ED_EE_test_dict = read_aggregated_test_file("./vanhell/ED1-EE2.csv")
    EE_ED_EE_test_dict = read_aggregated_test_file("./vanhell/EE1-ED2-EE3.csv")

    DD_test_lists = [DD_DE_DD_test_dict['D']['D'],
                     DE_DD_test_dict['D']['D']]
    gold_dict = {'DD':{},'DE':{},'EE':{},'ED':{}}
    for tl in DD_test_lists:
        for (c, r), f in Counter(tl).items():
            if r != "":
                if c not in gold_dict['DD']: gold_dict['DD'][c] = {}
                if r not in gold_dict['DD'][c]: gold_dict['DD'][c][r] = 0
                gold_dict['DD'][c][r] += f

    EE_test_lists = [EE_ED_EE_test_dict['E']['E'],
                     ED_EE_test_dict['E']['E']]
    for tl in EE_test_lists:
        for (c, r), f in Counter(tl).items():
            if r != "":
                if c not in gold_dict['EE']: gold_dict['EE'][c] = {}
                if r not in gold_dict['EE'][c]: gold_dict['EE'][c][r] = 0
                gold_dict['EE'][c][r] += f

    DE_test_lists = [DD_DE_DD_test_dict['D']['E'],
                     DE_DD_test_dict['D']['E']]
    for tl in DE_test_lists:
        for (c, r), f in Counter(tl).items():
            if r != "":
                if c not in gold_dict['DE']: gold_dict['DE'][c] = {}
                if r not in gold_dict['DE'][c]: gold_dict['DE'][c][r] = 0
                gold_dict['DE'][c][r] += f

    ED_test_lists = [EE_ED_EE_test_dict['E']['D'],
                     ED_EE_test_dict['E']['D']]
    for tl in ED_test_lists:
        for (c, r), f in Counter(tl).items():
            if r != "":
                if c not in gold_dict['ED']: gold_dict['ED'][c] = {}
                if r not in gold_dict['ED'][c]: gold_dict['ED'][c][r] = 0
                gold_dict['ED'][c][r] += f

    for d_of_d in gold_dict.values():
        for k,d in d_of_d.items():
            d_of_d[k] = {k: v for k, v in d.items() if v > threshold_frequency}

    return(gold_dict)

def print_results(tvds, rbds, apks, apks10, tvds2, rbds2, apks2, apks102):
    print("TVD MEANS: in-group: %.3f, out-group, %.3f" % (np.median([i[1] for i in tvds]), np.median([i[1] for i in tvds2])))
    #print(ttest_rel([i[1] for i in tvds], [j[1] for j in tvds2]))
    print(wilcoxon([i[1] for i in tvds], [j[1] for j in tvds2]))
    print("RBD MEANS: in-group: %.3f, out-group, %.3f" % (np.median([i[1] for i in rbds]), np.median([i[1] for i in rbds2])))
    #print(ttest_rel([i[1] for i in rbds], [j[1] for j in rbds2]))
    print(wilcoxon([i[1] for i in rbds], [j[1] for j in rbds2]))
    print("APK MEANS: in-group: %.3f, out-group, %.3f" % (np.median([i[1] for i in apks]), np.median([i[1] for i in apks2])))
    #print(ttest_rel([i[1] for i in apks], [j[1] for j in apks2]))
    print(wilcoxon([i[1] for i in apks], [j[1] for j in apks2]))
    print("APK (10) MEANS: in-group: %.3f, out-group, %.3f" % (np.median([i[1] for i in apks10]), np.median([i[1] for i in apks102])))
    #print(ttest_rel([i[1] for i in apks10], [j[1] for j in apks102]))
    print(wilcoxon([i[1] for i in apks10], [j[1] for j in apks102]))

if __name__ == "__main__":

    test_data_session = read_test_data('DD')

    test_data_aggregated = read_aggregated_test_data()['DD']
    norm_data_sf = read_norm_data("./SF_norms/sothflorida_complete.csv")
    norm_data_dutch = read_norm_data("./Dutch/shrunkdutch2.csv")
    #norm_data_eat = read_norm_data("./EAT/shrunkEAT.net_plain")

    # n = 30
    # greater_diff = {test_words[i]:(tvds2[i][1]-tvds[i][1]) for i in range(len(test_words)) if tvds2[i][1] > tvds[i][1]}
    # smaller_diff = {test_words[i]:(tvds2[i][1]-tvds[i][1]) for i in range(len(test_words)) if tvds2[i][1] <= tvds[i][1]}
    # print([k for k in sorted(greater_diff, key=greater_diff.get, reverse=True)][:n])
    # print([k for k in sorted(smaller_diff, key=smaller_diff.get)][:n])
    #
    # greater_diff = {test_words[i]:(rbds2[i][1]-rbds[i][1]) for i in range(len(test_words)) if rbds2[i][1] > rbds[i][1]}
    # smaller_diff = {test_words[i]:(rbds2[i][1]-rbds[i][1]) for i in range(len(test_words)) if rbds2[i][1] <= rbds[i][1]}
    # print([k for k in sorted(greater_diff, key=greater_diff.get, reverse=True)][:n])
    # print([k for k in sorted(smaller_diff, key=smaller_diff.get)][:n])
    #
    # greater_diff = {test_words[i]:(apks2[i][1]-apks[i][1]) for i in range(len(test_words)) if apks2[i][1] > apks[i][1]}
    # smaller_diff = {test_words[i]:(apks2[i][1]-apks[i][1]) for i in range(len(test_words)) if apks2[i][1] <= apks[i][1]}
    # print([k for k in sorted(greater_diff, key=greater_diff.get, reverse=True)][:n])
    # print([k for k in sorted(smaller_diff, key=smaller_diff.get)][:n])
    #
    # greater_diff = {test_words[i]:(apks102[i][1]-apks10[i][1]) for i in range(len(test_words)) if apks102[i][1] > apks[i][1]}
    # smaller_diff = {test_words[i]:(apks102[i][1]-apks10[i][1]) for i in range(len(test_words)) if apks102[i][1] <= apks[i][1]}
    # print([k for k in sorted(greater_diff, key=greater_diff.get, reverse=True)][:n])
    # print([k for k in sorted(smaller_diff, key=smaller_diff.get)][:n])


    # print(sorted([(tvds[idx][0], tvds2[idx][1] - tvds[idx][1]) for idx in range(len(tvds))], key=lambda x: -x[1])[:10])
    # print(sorted([(rbds[idx][0], rbds2[idx][1] - rbds[idx][1]) for idx in range(len(rbds))], key=lambda x: -x[1])[:10])
    # print(sorted([(apks[idx][0], apks2[idx][1] - apks[idx][1]) for idx in range(len(apks))], key=lambda x: -x[1])[:10])

    print("P1/1 & P1/2 vs. P1/1 & P2")
    test_words = sorted(list(set(test_data_session[2].keys()).intersection(set(test_data_session[0].keys()))))
    print(len(test_words), test_words)
    tvds, rbds, apks, apks10 = get_diff(test_data_session[0], test_data_session[1], test_words)
    tvds2, rbds2, apks2, apks102 = get_diff(test_data_session[0], test_data_session[2], test_words)
    print_results(tvds, rbds, apks, apks10, tvds2, rbds2, apks2, apks102)
    # tvds, rbds, apks = get_diff(test_data_session[0], test_data_session[1], test_words)
    # tvds2, rbds2, apks2 = get_diff(test_data_session[0], test_data_session[2], test_words)
    # print("TVD MEANS: in-group: %.3f, out-group, %.3f" % (np.mean([i[1] for i in tvds]), np.mean([i[1] for i in tvds2])))
    # print(ttest_rel([i[1] for i in tvds], [j[1] for j in tvds2]))
    # print("RBD MEANS: in-group: %.3f, out-group, %.3f" % (np.mean([i[1] for i in rbds]), np.mean([i[1] for i in rbds2])))
    # print(ttest_rel([i[1] for i in rbds], [j[1] for j in rbds2]))
    # print("APK MEANS: in-group: %.3f, out-group, %.3f" % (np.mean([i[1] for i in apks]), np.mean([i[1] for i in apks2])))
    # print(ttest_rel([i[1] for i in apks], [j[1] for j in apks2]))

    print("P1/1 & P2 vs. P1/1+2 & M(Dutch)")
    test_words = sorted(list(set(test_data_session[2].keys()).intersection(set(norm_data_dutch.keys()))))
    print(len(test_words), test_words)
    # if "trousers" in test_words: test_words.remove("trousers")
    tvds, rbds, apks, apks10 = get_diff(test_data_session[0], test_data_session[2], test_words)
    tvds2, rbds2, apks2, apks102 = get_diff(test_data_aggregated, norm_data_dutch, test_words)
    print_results(tvds, rbds, apks, apks10, tvds2, rbds2, apks2, apks102)

    print("P1/1 & P2 vs. P1/1+2 & M(English)")
    test_words = sorted(list(set(test_data_session[2].keys()).intersection(set(norm_data_sf.keys()))))
    print(len(test_words), test_words)
    # if "trousers" in test_words: test_words.remove("trousers")
    tvds, rbds, apks, apks10 = get_diff(test_data_session[0], test_data_session[2], test_words)
    tvds2, rbds2, apks2, apks102 = get_diff(test_data_aggregated, norm_data_sf, test_words)
    print_results(tvds, rbds, apks, apks10, tvds2, rbds2, apks2, apks102)

