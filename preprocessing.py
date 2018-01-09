import csv
import editdistance
import os
import pandas
import re
import sys
import utils
from collections import Counter
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from parameters import constants, parameters
from utils import invert_dict, normalize_dict

def activate_lemmatizers():
    global frog_installed, frog_lemmatizer, lemmas_nl, lemmas_nl_file, wn_lemmatizer

    wn_lemmatizer = WordNetLemmatizer()
    frog_installed = True
    with open("./data/lemmas_nl.csv", 'r') as lemmas_nl_file:
        lemmas_nl_df = pandas.read_csv(lemmas_nl_file, sep=",")
        lemmas_nl = dict(zip(lemmas_nl_df["word"], lemmas_nl_df["lemma"]))
    try:
        import frog
        frog_lemmatizer = frog.Frog(frog.FrogOptions(parser=False))
        lemmas_nl_file = open("./data/lemmas_nl.csv", 'a')
    except ImportError:
        frog_installed = False


def preprocess_orthography(fn, wordlist):
    with open(fn, 'w') as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["w1", "w2", "sim"])
        for idx1 in range(len(wordlist) - 1):
            w1 = wordlist[idx1]
            for idx2 in range(idx1 + 1, len(wordlist)):
                w2 = wordlist[idx2]
                sim = 1 - ((editdistance.eval(w1, w2)) / max(len(w1), len(w2)))
                if sim >= parameters["orthographic threshold"]:
                    writer.writerow([w1, w2, sim])


def read_orthography(fn, wordlist):
    if not os.path.isfile(fn):
        preprocess_orthography(fn, wordlist)
    df = pandas.read_csv(fn, sep=",", na_values="", keep_default_na=False)
    tuples = list(zip(df['w1'], df['w2'], df['sim']))
    orth_dict = {}
    for w1, w2, sim in tuples:
        if sim > parameters["orthographic threshold"]:
            orth_dict[(w1 + ":en", w2 + ":en")] = sim
            orth_dict[(w2 + ":en", w1 + ":en")] = sim
    return orth_dict


def preprocess_cognates(fn, en_nl_dic, words_nl, words_en):
    with open(fn, 'w') as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["nl", "en", "sim"])
        for word_en, wlist_nl in en_nl_dic.items():
            for word_nl in wlist_nl:
                if word_en in words_en and word_nl in words_nl:
                    sim = 1 - ((editdistance.eval(word_en, word_nl)) / max(len(word_en), len(word_nl)))
                    if sim >= parameters["cognate threshold"]:
                        writer.writerow([word_nl, word_en, sim])


def read_cognates(fn, en_nl_dic, words_nl, words_en):
    if not os.path.isfile(fn):
        preprocess_cognates(fn, en_nl_dic, words_nl, words_en)
    df = pandas.read_csv(fn, sep=",", na_values="", keep_default_na=False)
    tuples = list(zip(df['nl'], df['en'], df['sim']))
    cogn_dict = {}
    for w1, w2, sim in tuples:
        if sim > parameters["cognate threshold"]:
            cogn_dict[(w1 + ":nl", w2 + ":en")] = sim
            cogn_dict[(w2 + ":en", w2 + ":nl")] = sim
    return cogn_dict


def generate_ngrams(fn_list, wordlist):
    ngram_dict = {}
    unigram_dict = {}
    for fn in fn_list:
        with open(fn, 'r', encoding='ISO-8859-1') as f:
            test_reader = csv.reader(f, delimiter="\t")
            for row in test_reader:
                freq = int(row[0])
                w1 = lemmatize_word(row[1].lower(), "en", wn_lemmatizer)
                w2 = lemmatize_word(row[-1].lower(), "en", wn_lemmatizer)
                w1, w2 = sorted((w1, w2))

                if w1 in wordlist and w2 in wordlist and w1 not in stopwords.words('english') and w2 not in stopwords.words('english'):
                    if w1 not in unigram_dict:
                        unigram_dict[w1] = freq
                    else:
                        unigram_dict[w1] += freq
                    if w2 not in unigram_dict:
                        unigram_dict[w2] = freq
                    else:
                        unigram_dict[w2] += freq
                    if (w1, w2) not in ngram_dict:
                        ngram_dict[(w1, w2)] = freq
                    else:
                        ngram_dict[(w1, w2)] += freq

    return ngram_dict, unigram_dict


def preprocess_collocations(fn, wordlist):
    ngram_file_list = ["./data/coca/w2_.txt", "./data/coca/w3_.txt"]
    ngram_dict, unigram_dict = generate_ngrams(ngram_file_list, wordlist)
    probs = {}
    for tpl, freq in ngram_dict.items():
        for (w1, w2) in [(tpl[0], tpl[1]), (tpl[1], tpl[0])]:
            p_w2_given_w1 = freq / float(unigram_dict[w1])
            probs[(w1, w2)] = p_w2_given_w1
    with open(fn, 'w') as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["w1", "w2", "prob"])
        for tpl, p in sorted(probs.items()):
            row = [tpl[0], tpl[1], p]
            writer.writerow(row)


def read_cooccurences(fn, wordlist):
    if not os.path.isfile(fn):
        preprocess_collocations(fn, wordlist)
    df = pandas.read_csv(fn, sep=",", na_values="", keep_default_na=False)
    tuples = list(zip(df['w1'], df['w2'], df['prob']))
    coocc_dict = {}
    for w1, w2, sim in tuples:
        if sim > parameters["cut-off threshold"]:
            coocc_dict[(w1 + ":en", w2 + ":en")] = sim
    return coocc_dict


def read_alignments(fn, words_nl, words_en):
    df = pandas.read_csv(fn, sep=",", na_values="", keep_default_na=False)
    tuples = list(zip(df['en'], df['nl'], df['en-nl'], df['nl-en']))
    alignments = {}
    for w_en, w_nl, p_en_nl, p_nl_en in tuples:
        if w_en in words_en and w_nl in words_nl:
            if p_en_nl > parameters["cut-off threshold"]:
                alignments[(w_en + ":en", w_nl + ":nl")] = p_en_nl
            if p_nl_en > parameters["cut-off threshold"]:
                alignments[(w_nl + ":nl", w_en + ":en")] = p_nl_en
    return alignments


def lemmatize_word(word, lang, lemmatizer, pos=""):
    if lang == "nl":
        lemma = lemmatizer.process(word)[0]['lemma']
        print(word, lemma)
        sys.stdout.flush()
    elif lang == "en":
        if not pos:
            pos = pos_tag([word])[0][1]
        pos_wn = constants["wn mapping"].get(pos[0], wn.NOUN)
        lemma = lemmatizer.lemmatize(word, pos_wn)
    else:
        sys.exit("Unknown language in the lemmatization function!")
    return lemma


def filter_dict(cue_resp_dict):
    cue_resp_dict_filtered = {}
    for cue in cue_resp_dict:
        cue_resp_dict_filtered[cue] = {}
        for resp in cue_resp_dict[cue]:
            if resp not in constants["noise list"] and cue_resp_dict[cue][resp] > parameters["frequency threshold"]:
                cue_resp_dict_filtered[cue][resp] = cue_resp_dict[cue][resp]
    return cue_resp_dict_filtered


def get_cue_resp_dict_nl(fn):
    with open(fn, 'r') as norms:
        reader = csv.reader(norms, quotechar='"', quoting=csv.QUOTE_ALL, delimiter=";")
        next(reader, None)
        cue_resp_dict = {}
        for row in reader:
            cue = row[2].strip().lower()
            resp = row[3].strip().lower()
            if " " not in cue and " " not in resp and "," not in cue and "," not in resp:
                if frog_installed:
                    if cue in lemmas_nl:
                        cue = lemmas_nl[cue]
                    else:
                        cue_lemma = lemmatize_word(cue, "nl", frog_lemmatizer)
                        lemmas_nl[cue] = cue_lemma
                        lemmas_nl_file.write("%s,%s\n" % (cue, cue_lemma))
                        cue = cue_lemma
                    if resp in lemmas_nl:
                        resp = lemmas_nl[resp]
                    else:
                        resp_lemma = lemmatize_word(resp, "nl", frog_lemmatizer)
                        lemmas_nl[resp] = resp_lemma
                        lemmas_nl_file.write("%s,%s\n" % (resp, resp_lemma))
                        resp = resp_lemma
                else:
                    cue = lemmas_nl.get(cue) or cue
                    resp = lemmas_nl.get(resp) or resp
                if cue and resp:
                    if cue not in cue_resp_dict:
                        cue_resp_dict[cue] = {}
                    if resp not in cue_resp_dict[cue]:
                        cue_resp_dict[cue][resp] = 0.0
                    cue_resp_dict[cue][resp] += 1.0
    cue_resp_dict = filter_dict(cue_resp_dict)
    for cue in cue_resp_dict:
        cue_resp_dict[cue] = normalize_dict(cue_resp_dict[cue])
    return cue_resp_dict


def get_cue_resp_dict_en(dn):
    filenames = [dn + fn for fn in ["Cue_Target_Pairs.A-B", "Cue_Target_Pairs.C", "Cue_Target_Pairs.D-F", "Cue_Target_Pairs.G-K",
                 "Cue_Target_Pairs.L-O", "Cue_Target_Pairs.P-R", "Cue_Target_Pairs.S", "Cue_Target_Pairs.T-Z"]]
    wn_lemmatizer = WordNetLemmatizer()
    cue_resp_dict = {}
    for fn in filenames:
        with open(fn, 'r', encoding='ISO-8859-1') as norms:
            reader = csv.reader(norms, quotechar='"', quoting=csv.QUOTE_ALL, delimiter=",")
            next(reader, None)
            for row in reader:
                if " " not in row[0].strip() and " " not in row[1].strip():
                    cue = lemmatize_word(row[0].strip().lower(), "en", wn_lemmatizer)
                    resp = lemmatize_word(row[1].strip().lower(), "en", wn_lemmatizer)
                    freq = float(row[4].strip())
                    if cue not in cue_resp_dict:
                        cue_resp_dict[cue] = {}
                    if resp not in cue_resp_dict[cue]:
                        cue_resp_dict[cue][resp] = 0.0
                    cue_resp_dict[cue][resp] += freq
    cue_resp_dict = filter_dict(cue_resp_dict)
    for cue in cue_resp_dict:
        cue_resp_dict[cue] = normalize_dict(cue_resp_dict[cue])
    return cue_resp_dict


def read_norm_data(fn, lang):
    if not os.path.isfile(fn):
        preprocess_norms(lang)
    df = pandas.read_csv(fn, sep=",", na_values="", keep_default_na=False)
    tuples = list(zip(df['cue'], df['response'], df['p_response_given_cue']))
    norms_dict = {}
    for cue, resp, p in tuples:
        if cue+":"+lang not in norms_dict:
            norms_dict[cue+":"+lang] = {}
        if resp not in norms_dict[cue+":"+lang]:
            norms_dict[cue+":"+lang][resp+":"+lang] = 0
        norms_dict[cue+":"+lang][resp+":"+lang] += p
    return norms_dict


def preprocess_norms(lang):
    if lang == "nl":
        cue_resp_dict = get_cue_resp_dict_nl("./data/norms_nl/associationData.csv")
    elif lang == "en":
        cue_resp_dict = get_cue_resp_dict_en("./data/norms_en/")
    else:
        sys.exit("Unknown language in the preprocessing function!")
    with open("./data/norms_preprocessed_" + lang +".csv", 'w') as preprocessed_norms:
        writer = csv.writer(preprocessed_norms, quoting=csv.QUOTE_NONE, delimiter=",", escapechar='\\')
        writer.writerow(["cue", "response", "p_response_given_cue"])
        for cue in sorted(cue_resp_dict):
            for resp in sorted(cue_resp_dict[cue], key=cue_resp_dict[cue].get):
                writer.writerow([cue, resp, cue_resp_dict[cue][resp]])
    return cue_resp_dict


def clean_dict_entry(word):
    word = re.sub(r'(\([^)]*\))|(\[[^]]*\])|(\{[^}]*\})|(<[^>]*>)', '', word.strip()).strip()
    return word


def parse_dict_cc(fn):
    en_nl_dict = {}
    with open(fn, 'r') as dict_cc:
        reader = csv.reader(dict_cc, delimiter="\t")
        row = next(reader)
        while len(row) == 0 or row[0][0] == "#":
            row = next(reader)
        for row in reader:
            if len(row) == 3:
                word_en = row[0].strip()
                word_nl = row[1].strip()
                if " " not in word_en and " " not in word_nl and \
                                "," not in word_en and "," not in word_nl:
                    word_en = clean_dict_entry(word_en).lower()
                    word_nl = clean_dict_entry(word_nl).lower()
                    if word_en not in en_nl_dict:
                        en_nl_dict[word_en] = []
                    en_nl_dict[word_en].append(word_nl)
    return en_nl_dict


def parse_freedict(fn, invert=False):
    with open(fn, 'r') as freedict:
        translation_dict = {}
        entry_found = False
        word_source = ""
        for line in freedict:
            if line.strip() == "<entry>":
                entry_found = True
            elif line.strip() == "</entry>":
                entry_found = False
            if entry_found:
                if line.strip()[:6] == "<orth>":
                    word_source = re.sub(r'<orth>|</orth>', '', line.strip()).lower()
                    if " " in word_source:
                        entry_found = False
                    elif word_source not in translation_dict:
                        translation_dict[word_source] = []
                elif line.strip()[:7] == "<quote>":
                    word_target = re.sub(r'<quote>|</quote>', '', line.strip()).lower()
                    if " " not in word_target:
                        translation_dict[word_source].append(word_target)
    if invert:
        translation_dict = invert_dict(translation_dict)
    return translation_dict


def read_dict(fn, words_nl, words_en):
    if not os.path.isfile(fn):
        dict_cc = parse_dict_cc("./data/dict/dict.cc")
        free_dict1 = parse_freedict("./data/dict/eng-nld.tei")
        free_dict2 = parse_freedict("./data/dict/nld-eng.tei", invert=True)
        en_nl_dict = {}
        for word_en in set.union(set(dict_cc), set(free_dict1), set(free_dict2)):
            if word_en in words_en:
                en_nl_dict[word_en] = set((dict_cc.get(word_en) or []) +
                                          (free_dict1.get(word_en) or []) +
                                          (free_dict2.get(word_en) or []))
        with open(fn, 'w') as translation_dict:
            writer = csv.writer(translation_dict, quoting=csv.QUOTE_NONE, delimiter=",", escapechar='\\')
            writer.writerow(["en", "nl"])
            for word_en in sorted(en_nl_dict):
                lemma_en = lemmatize_word(word_en, "en", wn_lemmatizer)
                for word_nl in sorted(en_nl_dict[word_en]):
                    if word_nl in words_nl:
                        if frog_installed:
                            if word_nl in lemmas_nl:
                                lemma_nl = lemmas_nl[word_nl]
                            else:
                                lemma_nl = lemmatize_word(word_nl, "nl", frog_lemmatizer)
                                lemmas_nl[word_nl] = lemma_nl
                                lemmas_nl_file.write("%s,%s\n" % (word_nl, lemma_nl))
                        else:
                            lemma_nl = lemmas_nl.get(word_nl) or word_nl
                        writer.writerow([lemma_en, lemma_nl])
    else:
        en_nl_dict = {}
        with open(fn, 'r') as translation_dict:
            reader = csv.reader(translation_dict, quoting=csv.QUOTE_NONE, delimiter=",")
            next(reader)
            for row in reader:
                word_en = row[0]
                word_nl = row[1]
                if word_en not in en_nl_dict:
                    en_nl_dict[word_en] = []
                en_nl_dict[word_en].append(word_nl)
    return en_nl_dict


def read_bilingual_file(fn):
    # An auxiliary function that reads a single test file.
    conditions = fn.split("/")[-1].split('.')[0].split('-')
    cue_langs = [c[0] for c in conditions]
    cue_langs = [cue_langs[idx] + str(idx) for idx in range(len(cue_langs))]
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
                target_responses = []
                for r_idx in range(len(responses_mixed)):
                    if r_idx % n_conds == cond_idx:
                        resp_preprocessed = responses_mixed[r_idx]
                        if resp_preprocessed not in constants["noise list"]:
                            target_responses.append((cue, resp_preprocessed))
                resp_dict[cue_lang][target_lang].extend(target_responses)
    return resp_dict


def read_bilingual_data(condition):
    # Reads test data of Van Hell & De Groot (1998). CSV format needed.

    condition_lang = condition.split("-")[0]

    if condition_lang == "nl":
        DD_DE_DD_test_dict = read_bilingual_file("./data/bilingual/DD1-DE2-DD3.lemmas.csv")
        DE_DD_test_dict = read_bilingual_file("./data/bilingual/DE1-DD2.lemmas.csv")

        test_lists = [DD_DE_DD_test_dict['D0']['D0'],
                      DD_DE_DD_test_dict['D2']['D2'],
                      DE_DD_test_dict['D1']['D1']]

    elif condition_lang == "en":
        EE_ED_EE_test_dict = read_bilingual_file("./data/bilingual/EE1-ED2-EE3.lemmas.csv")
        ED_EE_test_dict = read_bilingual_file("./data/bilingual/ED1-EE2.lemmas.csv")

        test_lists = [EE_ED_EE_test_dict['E0']['E0'],
                      EE_ED_EE_test_dict['E2']['E2'],
                      ED_EE_test_dict['E1']['E1']]

    else:
        sys.exit("Condition unknown!")

    gold_dict = {}
    for session in range(len(test_lists)):
        for cue_resp, freq in Counter(test_lists[session]).items():
            cue = cue_resp[0] + ":" + condition_lang
            resp = cue_resp[1] + ":" + condition_lang
            if session not in gold_dict:
                gold_dict[session] = {}
            if cue not in gold_dict[session]:
                gold_dict[session][cue] = {}
            gold_dict[session][cue][resp] = freq
            if session < 2:
                if "aggregated" not in gold_dict:
                    gold_dict["aggregated"] = {}
                if cue not in gold_dict["aggregated"]:
                    gold_dict["aggregated"][cue] = {}
                if resp not in gold_dict["aggregated"][cue]:
                    gold_dict["aggregated"][cue][resp] = 0
                gold_dict["aggregated"][cue][resp] += freq

    for session in list(range(len(test_lists))) + ["aggregated"]:
        gold_dict[session] = filter_dict(gold_dict[session])
        for cue in gold_dict[session]:
            gold_dict[session][cue] = utils.normalize_dict(gold_dict[session][cue])

    return gold_dict
