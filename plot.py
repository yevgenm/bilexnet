from networks_igraph import *
from multiprocessing import Pool
from parameters import *

def main():

    #fn_en = "./EAT/shrunkEAT.net"
    fn_en = "./SF_norms/sothflorida_complete.csv"
    fn_nl = "./Dutch/shrunkdutch2.csv"

    en_nl_dic = utils.read_dict("./dict/dictionary.csv")
    nl_en_dic = utils.invert_dict(en_nl_dic)

    L1_assoc_coeff = 5
    L2_assoc_coeff = 5
    TE_coeff = 1
    orth_coeff = 2
    asymm_ratio = 5

    monoling = {"E": LexNetMo(fn=fn_en, language="en"),
                "D": LexNetMo(fn=fn_nl, language="nl")}
    mode = parameters["orth edge type"]
    biling = LexNetBi(fn_nl, fn_en, en_nl_dic, L1_assoc_coeff, L2_assoc_coeff, TE_coeff, orth_coeff, asymm_ratio, mode)

    gold_dict = read_test_data()

    test_wordlist = {"E": utils.filter_test_list(monoling["E"].G, sorted(gold_dict['EE'].keys())),
                     "D": utils.filter_test_list(monoling["D"].G, sorted(gold_dict['DD'].keys()))}

    test = ['EE']

    for test_condition in test:

        cue_lang = test_condition[0]
        target_lang = test_condition[1]

        test_list = test_wordlist[cue_lang]

        l = ["duty", "cause", "opportunity", "attempt", "ease", "revenge", "truth", "conscience", "memory", "faith", "demand", "possession", "care", "advantage", "shop", "mirror", "rifle", "potato", "knife", "bottle", "skirt", "flower", "tree", "farm", "hospital", "bird", "bike", "jail", "insight", "chance", "shame", "plan", "motive", "block", "quality", "hell", "figure", "method", "principle", "information", "metal", "circle", "panic", "shoulder", "season", "finger", "captain", "daughter", "pepper", "slave", "apple", "snow", "winter", "coffee", "rose", "police", "train", "doctor"]
        plot_list = [i+":EN" for i in l]
        for w in plot_list:
            monoling[cue_lang].plot_subgraph(w, parameters["baseline depth"], "mon")
            biling.plot_subgraph(w, parameters["model depth"], "biling")

main()
