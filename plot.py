from networks_igraph import *
from multiprocessing import Pool
from parameters import *

def main():

    #fn_en = "./EAT/shrunkEAT.net"
    fn_en = "./SF_norms/sothflorida_complete.csv"
    fn_nl = "./Dutch/shrunkdutch2.csv"

    en_nl_dic = utils.read_dict("./dict/dictionary.csv")
    nl_en_dic = utils.invert_dict(en_nl_dic)

    TE_assoc_ratio = 300
    orth_assoc_ratio = 5
    asymm_ratio = 1

    monoling = {"E": LexNetMo(fn=fn_en, language="en"),
                "D": LexNetMo(fn=fn_nl, language="nl")}
    biling = LexNetBi(fn_nl, fn_en, en_nl_dic, TE_assoc_ratio, orth_assoc_ratio, asymm_ratio)

    gold_dict = read_test_data()

    test_wordlist = {"E": utils.filter_test_list(monoling["E"].G, sorted(gold_dict['EE'].keys())),
                     "D": utils.filter_test_list(monoling["D"].G, sorted(gold_dict['DD'].keys()))}

    test = ['EE']

    for test_condition in test:

        cue_lang = test_condition[0]
        target_lang = test_condition[1]

        test_list = test_wordlist[cue_lang]

        plot_list = ["slave:EN"]
        for w in plot_list:
            monoling[target_lang].plot_subgraph(w, parameters["baseline depth"], "en")
            biling.plot_subgraph(w, parameters["model depth"], "biling")

main()
