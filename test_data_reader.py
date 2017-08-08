import csv
from parameters import parameters, extras
from collections import Counter

def read_test_file(fn):
    # An auxiliary function that reads a single test file.
    conditions = fn.split("/")[-1].split('.')[0].split('-')
    cue_langs = [c[0] for c in conditions]
    target_langs = [c[1] for c in conditions]
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
                target_responses = [(cue+extras["language mapping"][cue_lang], responses_mixed[r_idx]+extras["language mapping"][target_lang]) for r_idx in range(len(responses_mixed))
                                    if r_idx%n_conds==cond_idx and responses_mixed[r_idx] not in extras["noise list"] ]
                resp_dict[cue_lang][target_lang].extend(target_responses)
    return(resp_dict)

def read_test_data():
    # Reads test data of Van Hell & De Groot (1998). CSV format needed.

    DD_DE_DD_test_dict = read_test_file("./vanhell/DD1-DE2-DD3.lemmas.csv")
    DE_DD_test_dict = read_test_file("./vanhell/DE1-DD2.lemmas.csv")
    ED_EE_test_dict = read_test_file("./vanhell/ED1-EE2.lemmas.csv")
    EE_ED_EE_test_dict = read_test_file("./vanhell/EE1-ED2-EE3.lemmas.csv")

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
            d_of_d[k] = {k: v for k, v in d.items() if v > parameters["frequency threshold"]}

    return(gold_dict)
