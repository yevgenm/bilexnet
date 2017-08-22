mapping = {'E': ":EN", 'D': ":NL"}
import operator
import math
import sys
from collections import Counter
from scipy.stats import ttest_rel
from utils import get_rbd
import csv
import numpy as np
import ml_metrics as metrics
import pandas
import pickle
from scipy.stats import wilcoxon
from parameters import parameters
import utils

en_nl_dic = utils.read_dict("./dict/dictionary.csv")

nouns_en = ["duty","cause","opportunity","favor","attempt","ease","revenge","truth","conscience","memory","faith","demand","possession","care","advantage","shop","mirror","rifle","potato","knife","bottle","skirt","flower","tree","trouser","farm","hospital","bird","bike","jail","insight","chance","shame","plan","motive","block","quality","hell","figure","method","principle","information","metal","circle","panic","shoulder","season","finger","captain","daughter","pepper","slave","apple","snow","winter","coffee","rose","police","train","doctor"]
nouns_nl = ["plicht","oorzaak","gelegenheid","gunst","poging","gemak","wraak","waarheid","geweten","geheugen","geloof","eis","bezit","zorg","voordeel","winkel","spiegel","geweer","aardappel","mes","fles","rok","bloem","boom","broek","boerderij","ziekenhuis","vogel","fiets","gevangenis","inzicht","kans","schaamte","plan","motief","blok","kwaliteit","hel","figuur","methode","principe","informatie","metaal","cirkel","paniek","schouder","seizoen","vinger","kapitein","dochter","peper","slaaf","appel","sneeuw","winter","koffie","roos","politie","trein","dokter"]
n_abs_ncog = sorted(["duty","opportunity","favor","attempt","advantage","ease","revenge","truth","conscience","cause","memory","demand","care","faith","possession"])
n_abs_cog = sorted(["insight","shame","plan","motive","quality","hell","block","method","chance","principle","information","metal","circle","panic","figure"])
n_con_ncog = sorted(["shop","mirror","rifle","potato","knife","bottle","skirt","flower","tree","hospital","trousers","farm","bird","bike","jail"])
n_con_cog = sorted(["shoulder","season","finger","captain","daughter","pepper","slave","apple","snow","winter","coffee","rose","police","train","doctor"])
v_abs_ncog = sorted(["refuse","admit","disturb","succeed","guess","promise","understand"])
v_abs_cog = sorted(["forget","bend","dare","arrest","hate","hope","spread","irritate"])
v_con_ncog = sorted(["tremble","move","baptize","throw","calculate","marry","cry","paint"])
v_con_cog = sorted(["sneeze","climb","frown","swim","listen","steal","sing"])
nouns = n_abs_ncog + n_abs_cog + n_con_ncog + n_con_cog
verbs = v_abs_ncog + v_abs_cog + v_con_ncog + v_con_cog
abst = n_abs_ncog + n_abs_cog + v_abs_ncog + v_abs_cog
con = n_con_ncog + n_con_cog + v_con_ncog + v_con_cog
ncog = n_abs_ncog + n_con_ncog + v_abs_ncog + v_con_ncog
cog = n_abs_cog + n_con_cog + v_abs_cog + v_con_cog

np.random.seed(12345678)
threshold_frequency = 1
noise_list = ["x", "", None]

def read_norm_data(fn):
    # Constructs the big graph from CSV.
    df = pandas.read_csv(fn, sep=";", na_values="", keep_default_na=False)
    tuples = Counter()
    for i in range(1,2):
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
    raw = math.fsum(d.values())
    if raw == 0: return {}
    factor = target / raw
    for k in d:
        d[k] = d[k]*factor
    key_for_max = max(d.items(), key=operator.itemgetter(1))[0]
    diff = 1.0 - math.fsum(d.values())
    #print("discrepancy = " + str(diff))
    d[key_for_max] += diff
    return d

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
        DD_DE_DD_test_dict = read_test_file("./vanhell/DD1-DE2-DD3.lemmas.csv")
        DE_DD_test_dict = read_test_file("./vanhell/DE1-DD2.lemmas.csv")

        test_lists = [DD_DE_DD_test_dict['D0']['D0'],
                          DD_DE_DD_test_dict['D2']['D2'],
                          DE_DD_test_dict['D1']['D1']]

    elif condition[1] == "E":
        EE_ED_EE_test_dict = read_test_file("./vanhell/EE1-ED2-EE3.lemmas.csv")
        ED_EE_test_dict = read_test_file("./vanhell/ED1-EE2.lemmas.csv")

        test_lists = [EE_ED_EE_test_dict['E0']['E0'],
                      EE_ED_EE_test_dict['E2']['E2'],
                      ED_EE_test_dict['E1']['E1']]
    else:
        sys.exit("Condition unknown!")

    gold_dict = {}
    for session in range(len(test_lists)):
        for k,v in Counter(test_lists[session]).items():
            if session not in gold_dict:
                gold_dict[session] = {}
            if k[0] not in gold_dict[session]:
                gold_dict[session][k[0]] = {}
            if v > threshold_frequency and k[1] not in noise_list:
                gold_dict[session][k[0]][k[1]] = v
        for d in gold_dict[session]:
            gold_dict[session][d] = normalize_dict(gold_dict[session][d])

    return(gold_dict)


def get_diff(d1, d2, test_words):
    tvds = []
    rbds = []
    jacs = []
    apks = []
    apks10 = []
    for w in sorted(test_words):
        k = min(len(d1[w]), len(d2[w]))
        tw1_sorted = sorted(d1[w].items(), key=lambda x: (-x[1], x[0]))
        tw2_sorted = sorted(d2[w].items(), key=lambda x: (-x[1], x[0]))
        lw1 = [e[0] for e in tw1_sorted]
        lw2 = [e[0] for e in tw2_sorted]
        dw1 = normalize_dict(dict(tw1_sorted[:k]))
        dw2 = normalize_dict(dict(tw2_sorted[:k]))
        sw1 = set(lw1[:parameters["jaccard k"]])
        sw2 = set(lw2[:parameters["jaccard k"]])
        tvd = 0.5 * sum(abs((dw1.get(resp) or 0) - (dw2.get(resp) or 0)) for resp in sorted(set(dw1) | set(dw2)))
        rbd = get_rbd(lw1, lw2)
        jac = 1 - len(set.intersection(sw1,sw2))/float(len(set.union(sw1,sw2)))

        sl, ll = sorted([(len(lw1), lw1), (len(lw2), lw2)])
        s, S = sl
        l, L = ll
        apk = 1 - metrics.apk(S, L, s)
        apk10 = 1 - metrics.apk(S[:parameters["apk k"]], L, parameters["apk k"])

        tvds.append((w, tvd))
        rbds.append((w, rbd))
        jacs.append((w, jac))
        apks.append((w, apk))
        apks10.append((w, apk10))
    return tvds, rbds, jacs, apks, apks10

def get_diff_words(d_bil_en, d_en, d_bil_nl, d_nl, test_words_en, test_words_nl):
    f = open("comparison_log_new.csv", 'w')
    fwriter = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    fwriter.writerow(["Cue", "Response", "Response rank in mon. EN", "Response TE and its rank in bil. DU", "Response TE and its rank in mon. DU"])
    for idx in range(len(test_words)):
        cue_en = test_words_en[idx]
        cue_nl = test_words_nl[idx]
        if cue_en in d_en and cue_en in d_bil_en and cue_nl in d_bil_nl and cue_nl in d_nl:
            tw1_sorted = sorted(d_bil_en[cue_en].items(), key=lambda x: (-x[1], x[0]))
            tw2_sorted = sorted(d_en[cue_en].items(), key=lambda x: (-x[1], x[0]))
            tw3_sorted = sorted(d_bil_nl[cue_nl].items(), key=lambda x: (-x[1], x[0]))
            tw4_sorted = sorted(d_nl[cue_nl].items(), key=lambda x: (-x[1], x[0]))
            lw1 = [e[0] for e in tw1_sorted]
            lw2 = [e[0] for e in tw2_sorted]
            lw3 = [e[0] for e in tw3_sorted]
            lw4 = [e[0] for e in tw4_sorted]
            sw1 = set(lw1[:parameters["jaccard k"]])
            sw2 = set(lw2[:parameters["jaccard k"]])
            sw2_full = set(lw2)
            #sw3 = set(lw3[:parameters["jaccard k"]])
            sw3 = set(lw3)
            sw4 = set(lw4)
            for w in sw1-sw2:
                row = []
                row.append(cue_en + " (" + cue_nl + ")")
                row.append(w)
                row.append(lw2.index(w)+1 if w in lw2 else "NA")
                translations = en_nl_dic.get(w+":EN")
                if not translations: row.append("MISSING")
                else:
                    t_best, pos_best = ("NA", 666)
                    for t in translations:
                        if t[:-3] in lw3:
                            if lw3.index(t[:-3]) + 1 < pos_best:
                                t_best, pos_best = (t[:-3], lw3.index(t[:-3]) + 1)
                    if pos_best < 666:
                        row.append(t_best + ", " + str(pos_best))
                    else:
                        row.append("NA")
                if not translations: row.append("MISSING")
                else:
                    t_best, pos_best = ("NA", 666)
                    for t in translations:
                        if t[:-3] in lw4:
                            if lw4.index(t[:-3]) + 1 < pos_best:
                                t_best, pos_best = (t[:-3], lw4.index(t[:-3]) + 1)
                    if pos_best < 666:
                        row.append(t_best + ", " + str(pos_best))
                    else:
                        row.append("NA")
                fwriter.writerow(row)
        # print(w, ":\t", ",".join(sw2), "\tvs.\t", ",".join(sw1), "[", w2, ":", ",".join(sw3), "]")
    f.close()

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
                                    if r_idx%n_conds==cond_idx and preprocess_word(responses_mixed[r_idx]) not in noise_list]
                resp_dict[cue_lang][target_lang].extend(target_responses)
    return(resp_dict)

def read_aggregated_test_data():
    # Reads test data of Van Hell & De Groot (1998). CSV format needed.

    DD_DE_DD_test_dict = read_aggregated_test_file("./vanhell/DD1-DE2-DD3.lemmas.csv")
    DE_DD_test_dict = read_aggregated_test_file("./vanhell/DE1-DD2.lemmas.csv")
    ED_EE_test_dict = read_aggregated_test_file("./vanhell/ED1-EE2.lemmas.csv")
    EE_ED_EE_test_dict = read_aggregated_test_file("./vanhell/EE1-ED2-EE3.lemmas.csv")

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

def print_results(tvds, rbds, jacs, apks, apks10, tvds2, rbds2, jacs2, apks2, apks102):
    print("RBD MEANS: in-group: %.3f, out-group, %.3f" % (np.mean([i[1] for i in rbds]), np.mean([i[1] for i in rbds2])))
    #print(ttest_rel([i[1] for i in rbds], [j[1] for j in rbds2]))
    print(wilcoxon([i[1] for i in rbds], [j[1] for j in rbds2]))
    print("TVD MEANS: in-group: %.3f, out-group, %.3f" % (np.mean([i[1] for i in tvds]), np.mean([i[1] for i in tvds2])))
    #print(ttest_rel([i[1] for i in tvds], [j[1] for j in tvds2]))
    print(wilcoxon([i[1] for i in tvds], [j[1] for j in tvds2]))
    print("JAC MEANS: in-group: %.3f, out-group, %.3f" % (np.mean([i[1] for i in jacs]), np.mean([i[1] for i in jacs2])))
    #print(ttest_rel([i[1] for i in jacs], [j[1] for j in jacs2]))
    print(wilcoxon([i[1] for i in jacs], [j[1] for j in jacs2]))
    print("APK MEANS: in-group: %.3f, out-group, %.3f" % (np.mean([i[1] for i in apks]), np.mean([i[1] for i in apks2])))
    #print(ttest_rel([i[1] for i in apks], [j[1] for j in apks2]))
    print(wilcoxon([i[1] for i in apks], [j[1] for j in apks2]))
    print("APK (10) MEANS: in-group: %.3f, out-group, %.3f" % (np.mean([i[1] for i in apks10]), np.mean([i[1] for i in apks102])))
    #print(ttest_rel([i[1] for i in apks10], [j[1] for j in apks102]))
    print(wilcoxon([i[1] for i in apks10], [j[1] for j in apks102]))

if __name__ == "__main__":

    condition_en = "EE"
    test_data_session_en = read_test_data(condition_en)
    test_data_aggregated_en = read_aggregated_test_data()[condition_en]

    condition_nl = "DD"
    test_data_session_nl = read_test_data(condition_nl)
    test_data_aggregated_nl = read_aggregated_test_data()[condition_nl]

    norm_data_sf = read_norm_data("./SF_norms/sothflorida_complete.csv")
    norm_data_dutch = read_norm_data("./Dutch/dutch_final.csv")
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
    test_words = sorted(list(set(test_data_session_en[2].keys()).intersection(set(test_data_session_en[0].keys()))))
    print(len(test_words), test_words)
    tvds, rbds, jacs, apks, apks10 = get_diff(test_data_session_en[0], test_data_session_en[1], test_words)
    tvds2, rbds2, jacs2, apks2, apks102 = get_diff(test_data_session_en[0], test_data_session_en[2], test_words)
    print_results(tvds, rbds, jacs, apks, apks10, tvds2, jacs2, rbds2, apks2, apks102)
    # tvds, rbds, apks = get_diff(test_data_session[0], test_data_session[1], test_words)
    # tvds2, rbds2, apks2 = get_diff(test_data_session[0], test_data_session[2], test_words)
    # print("TVD MEANS: in-group: %.3f, out-group, %.3f" % (np.mean([i[1] for i in tvds]), np.mean([i[1] for i in tvds2])))
    # print(ttest_rel([i[1] for i in tvds], [j[1] for j in tvds2]))
    # print("RBD MEANS: in-group: %.3f, out-group, %.3f" % (np.mean([i[1] for i in rbds]), np.mean([i[1] for i in rbds2])))
    # print(ttest_rel([i[1] for i in rbds], [j[1] for j in rbds2]))
    # print("APK MEANS: in-group: %.3f, out-group, %.3f" % (np.mean([i[1] for i in apks]), np.mean([i[1] for i in apks2])))
    # print(ttest_rel([i[1] for i in apks], [j[1] for j in apks2]))

    print("P1/1 & P2 vs. P1/1+2 & M(Dutch)")
    test_words = sorted(list(set(test_data_session_nl[2].keys()).intersection(set(norm_data_dutch.keys()))))
    print(len(test_words), test_words)
    tvds, rbds, jacs, apks, apks10 = get_diff(test_data_session_nl[0], test_data_session_nl[2], test_words)
    tvds2, rbds2, jacs2, apks2, apks102 = get_diff(test_data_aggregated_nl, norm_data_dutch, test_words)
    print_results(tvds, rbds, jacs, apks, apks10, tvds2, rbds2, jacs2, apks2, apks102)

    print("P1/1 & P2 vs. P1/1+2 & M(English)")
    test_words = sorted(list(set(test_data_session_en[2].keys()).intersection(set(norm_data_sf.keys()))))
    print("Missing words" + str([set(test_data_session_en[2].keys())-set(norm_data_sf.keys())]))
    # test_words = [i for i in test_words if i in abst]
    print(len(test_words), test_words)
    tvds, rbds, jacs, apks, apks10 = get_diff(test_data_session_en[0], test_data_session_en[2], test_words)
    tvds2, rbds2, jacs2, apks2, apks102 = get_diff(test_data_aggregated_en, norm_data_sf, test_words)
    print_results(tvds, rbds, jacs, apks, apks10, tvds2, rbds2, jacs2, apks2, apks102)

    #test_words_en = nouns_en
    #test_words_nl = nouns_nl
    #get_diff_words(test_data_aggregated_en, norm_data_sf, test_data_aggregated_nl, norm_data_dutch, test_words_en, test_words_nl)
