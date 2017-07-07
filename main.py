from networks_igraph import *
from multiprocessing import Pool
from parameters import *

res_dir = "./results_test_lemmatized/"

def test_model(args):

    test_list, test_condition, fn_nl, fn_en, en_nl_dic, tvd_base, rbd_base, apk_base, gold_dict, assoc_coeff, TE_coeff, orth_coeff, asymm_ratio = args
    log_fn = res_dir + "log_" + test_condition + "_assoc_" + str(assoc_coeff) + "_TE_" + str(TE_coeff) + "_orth_" + str(orth_coeff) + "_asymm_" + str(asymm_ratio)
    log_file = open(log_fn, "w")
    biling = LexNetBi(fn_nl, fn_en, en_nl_dic, assoc_coeff, TE_coeff, orth_coeff, asymm_ratio)

    log_file.write("NET:%s, DEPTH:%i, ASSOC: %i, TE:%i, ORTH:%i, ASYMM: %i\n" % ("bi", parameters["model depth"], assoc_coeff, TE_coeff, orth_coeff, asymm_ratio))
    tvd_m, rbd_m, apk_m = biling.test_network(test_list, parameters["model depth"], test_condition, gold_full=gold_dict, verbose=True, log_file=log_file)

    log_file.write("TVD t-test: T=%.2f, p=%.3f\n" % (ttest_rel(tvd_base, tvd_m)[0], ttest_rel(tvd_base, tvd_m)[1]))
    log_file.write("RBD t-test: T=%.2f, p=%.3f\n" % (ttest_rel(rbd_base, rbd_m)[0], ttest_rel(rbd_base, rbd_m)[1]))
    log_file.write("APK t-test: T=%.2f, p=%.3f\n" % (ttest_rel(apk_base, apk_m)[0], ttest_rel(apk_base, apk_m)[1]))

    log_file.close()

    return(assoc_coeff, TE_coeff, orth_coeff, asymm_ratio, tvd_m, rbd_m, apk_m)


def main():

    #fn_en = "./EAT/shrunkEAT.net"
    if os.path.exists(res_dir):
        if os.listdir(res_dir) != []:
            sys.exit("Result directory not empty!")
    else:
        os.makedirs(res_dir)

    fn_en = "./SF_norms/sothflorida_complete.csv"
    fn_nl = "./Dutch/shrunkdutch2.csv"

    en_nl_dic = utils.read_dict("./dict/dictionary.csv")
    nl_en_dic = utils.invert_dict(en_nl_dic)

    monoling = {"E": LexNetMo(fn=fn_en, language="en"),
                "D": LexNetMo(fn=fn_nl, language="nl")}

    gold_dict = read_test_data()

    test_wordlist = {"E": utils.filter_test_list(monoling["E"].G, sorted(gold_dict['EE'].keys())),
                     "D": utils.filter_test_list(monoling["D"].G, sorted(gold_dict['DD'].keys()))}

    test = ['EE']

    for test_condition in test:

        cue_lang = test_condition[0]
        target_lang = test_condition[1]

        test_list = test_wordlist[cue_lang]

        if cue_lang==target_lang:
            if os.path.isfile(res_dir + "log_baseline_"+test_condition):
                tvd_base, rbd_base, apk_base = monoling[target_lang].test_network(test_list, parameters["baseline depth"], test_condition, gold_full=gold_dict, verbose=False)
            else:
                log_base_file = open(res_dir + "log_baseline_"+test_condition, 'w')
                log_base_file.write("NET:%s, DEPTH:%i\n" % (target_lang, parameters["baseline depth"]))
                tvd_base, rbd_base, apk_base = monoling[target_lang].test_network(test_list, parameters["baseline depth"], test_condition, gold_full=gold_dict, verbose=True, log_file=log_base_file)
                log_base_file.close()
        else:
            tvd_base = rbd_base = apk_base = [0]*len(test_list)

        log_per_word = open(res_dir + "log_per_word_"+test_condition+".tsv", 'w')
        log_per_word.write("assoc\tTE\torth\tasymm\t")
        for w in test_list:
            log_per_word.write("%s (tvd)\t%s (rbd)\t%s (apk)\t" % (w, w, w))

        log_per_word.write("\n")
        log_per_word.flush()

        meta_args = [test_list, test_condition, fn_nl, fn_en, en_nl_dic, tvd_base, rbd_base, apk_base, gold_dict]

        par = [ [assoc_coeff, TE_coeff, orth_coeff, asymm_ratio]
                for assoc_coeff in [0, 1, 2, 3, 5, 10, 20, 50, 100, 500]
                for TE_coeff in [1, 2, 3, 5, 10]
                for orth_coeff in [0, 1, 2, 3, 5, 10]
                for asymm_ratio in [1, 2, 3, 5, 10]
                if assoc_coeff==1 or not (assoc_coeff==TE_coeff and TE_coeff==orth_coeff) ]


        args = [meta_args + par_set for par_set in par]

        #for a in args:
        #    run_test(a)

        pool = Pool(5)
        for assoc_coeff, TE_coeff, orth_coeff, asymm_ratio, tvd_m, rbd_m, apk_m in pool.imap(test_model, args):
            log_per_word.write("%d\t%d\t%d\t%d\t" % (assoc_coeff, TE_coeff, orth_coeff, asymm_ratio))
            for idx in range(len(test_list)):
                log_per_word.write("%.3f\t%.3f\t%.3f\t" % (tvd_m[idx] - tvd_base[idx], rbd_m[idx] - rbd_base[idx], apk_m[idx] - apk_base[idx]))
            log_per_word.write("\n")
            log_per_word.flush()

        log_per_word.close()

        # plot_list = ["yellow:EN"]
        # for w in plot_list:
        #     plot_subgraph(en, w, 3, "en")
        #     plot_subgraph(biling, w, 3, "biling")

main()
