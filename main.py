from networks_igraph import *
from multiprocessing import Pool
from parameters import *

def test_model(args):

    test_list, test_condition, fn_nl, fn_en, en_nl_dic, tvd_base, rbd_base, apk_base, apk_base_10, gold_dict, res_dir_cond, L1_assoc_coeff, L2_assoc_coeff, TE_coeff, orth_coeff, asymm_ratio = args
    log_fn = res_dir_cond + "log_" + test_condition + "_L1_assoc_" + str(L1_assoc_coeff) + "_L2_assoc_" + str(
        L2_assoc_coeff) + "_TE_" + str(TE_coeff) + "_orth_" + str(orth_coeff) + "_asymm_" + str(asymm_ratio)
    log_file = open(log_fn, "w")
    biling = LexNetBi(fn_nl, fn_en, en_nl_dic, L1_assoc_coeff, L2_assoc_coeff, TE_coeff, orth_coeff, asymm_ratio)

    if test_condition == "ED":
        dict_for_filtering = en_nl_dic
    elif test_condition == "DE":
        dict_for_filtering = utils.invert_dict(en_nl_dic)
    else:
        dict_for_filtering = None

    log_file.write("NET:%s, DEPTH:%i, L1 ASSOC: %i, L2 ASSOC: %i, TE:%i, ORTH:%i, ASYMM: %i\n" % ("bi", parameters[
        "model depth"], L1_assoc_coeff, L2_assoc_coeff, TE_coeff, orth_coeff, asymm_ratio))
    tvd_m, rbd_m, apk_m, apk_m_10 = biling.test_network(test_list, parameters["model depth"], test_condition, translation_dict=dict_for_filtering, gold_full=gold_dict, verbose=True, log_file=log_file)

    log_file.write("TVD t-test: T=%.2f, p=%.3f\n" % (ttest_rel(tvd_base, tvd_m)[0], ttest_rel(tvd_base, tvd_m)[1]))
    log_file.write("RBD t-test: T=%.2f, p=%.3f\n" % (ttest_rel(rbd_base, rbd_m)[0], ttest_rel(rbd_base, rbd_m)[1]))
    log_file.write("APK(k) t-test: T=%.2f, p=%.3f\n" % (ttest_rel(apk_base, apk_m)[0], ttest_rel(apk_base, apk_m)[1]))
    log_file.write("APK(10) t-test: T=%.2f, p=%.3f\n" % (ttest_rel(apk_base_10, apk_m_10)[0], ttest_rel(apk_base_10, apk_m_10)[1]))

    log_file.close()

    return(L1_assoc_coeff, L2_assoc_coeff, TE_coeff, orth_coeff, asymm_ratio, tvd_m, rbd_m, apk_m, apk_m_10)


def main():

    res_dir = sys.argv[1]
    if res_dir[-1] != "/":
        res_dir += "/"
    workers = int(sys.argv[2])

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    fn_en = "./SF_norms/sothflorida_complete.csv"
    fn_nl = "./Dutch/shrunkdutch2.csv"

    en_nl_dic = utils.read_dict("./dict/dictionary.csv")

    monoling = {"E": LexNetMo(fn=fn_en, language="en"),
                "D": LexNetMo(fn=fn_nl, language="nl")}

    gold_dict = read_test_data()

    test_wordlist = {"E": utils.filter_test_list(monoling["E"].G, sorted(gold_dict['EE'].keys())),
                     "D": utils.filter_test_list(monoling["D"].G, sorted(gold_dict['DD'].keys()))}

    test = ['EE']

    for test_condition in test:

        res_dir_cond = res_dir + test_condition + "/"
        if os.path.exists(res_dir_cond):
            if os.listdir(res_dir_cond) != []:
                sys.exit("Result directory not empty!")
        else:
            os.makedirs(res_dir_cond)

        cue_lang = test_condition[0]
        target_lang = test_condition[1]

        test_list = test_wordlist[cue_lang]

        if cue_lang==target_lang:
            if os.path.isfile(res_dir_cond + "log_baseline_"+test_condition):
                tvd_base, rbd_base, apk_base, apk_base_10 = monoling[target_lang].test_network(test_list, parameters["baseline depth"], test_condition, gold_full=gold_dict, verbose=False)
            else:
                log_base_file = open(res_dir_cond + "log_baseline_"+test_condition, 'w')
                log_base_file.write("NET:%s, DEPTH:%i\n" % (target_lang, parameters["baseline depth"]))
                tvd_base, rbd_base, apk_base, apk_base_10 = monoling[target_lang].test_network(test_list, parameters["baseline depth"], test_condition, gold_full=gold_dict, verbose=True, log_file=log_base_file)
                log_base_file.close()
        else:
            tvd_base = rbd_base = apk_base = apk_base_10 = [0]*len(test_list)

        log_per_word = open(res_dir_cond + "log_per_word_"+test_condition+".tsv", 'w')
        log_per_word.write("L1_assoc\tL2_assoc\tTE\torth\tasymm\t")
        for w in test_list:
            log_per_word.write("%s (tvd)\t%s (rbd)\t%s (apk_k)\t%s (apk_10)\t" % (w, w, w, w))

        log_per_word.write("\n")
        log_per_word.flush()

        meta_args = [test_list, test_condition, fn_nl, fn_en, en_nl_dic, tvd_base, rbd_base, apk_base, apk_base_10, gold_dict, res_dir_cond]

        par = [ [L1_assoc_coeff, L2_assoc_coeff, TE_coeff, orth_coeff, asymm_ratio]
                # for L1_assoc_coeff in [1, 2, 3, 5, 7, 10, 20]
                # for L2_assoc_coeff in [0, 1, 2, 3, 5, 7, 10, 20]
                # for TE_coeff in [1, 2, 3, 5, 7, 10]
                # for orth_coeff in [0, 1, 2, 3, 5, 7, 10]
                # for asymm_ratio in [1, 2, 3, 5, 7, 10]
                # if L1_assoc_coeff==1 or not (L1_assoc_coeff==TE_coeff==L2_assoc_coeff==orth_coeff)
                for (L1_assoc_coeff, L2_assoc_coeff, TE_coeff, orth_coeff, asymm_ratio) in [(2,2,1,1,2)]
                # for (assoc_coeff, TE_coeff, orth_coeff, asymm_ratio) in
                # zip([1,1,1,1,1,1,2,2,2,3,3,5,5,20],[1]*14,[0,1,2,3,0,1,0,1,2,0,2,0,1,0],[1,1,1,1,2,2,1,1,1,1,1,1,1,1])
              ]


        args = [meta_args + par_set for par_set in par]

        #for a in args:
        #    run_test(a)

        pool = Pool(workers)
        for L1_assoc_coeff, L2_assoc_coeff, TE_coeff, orth_coeff, asymm_ratio, tvd_m, rbd_m, apk_m, apk_m_10 in pool.imap(test_model, args):
            log_per_word.write("%d\t%d\t%d\t%d\t%d\t" % (L1_assoc_coeff, L2_assoc_coeff, TE_coeff, orth_coeff, asymm_ratio))
            for idx in range(len(test_list)):
                log_per_word.write("%.3f\t%.3f\t%.3f\t%.3f\t" % (tvd_m[idx] - tvd_base[idx], rbd_m[idx] - rbd_base[idx], apk_m[idx] - apk_base[idx], apk_m_10[idx] - apk_base_10[idx]))
            log_per_word.write("\n")
            log_per_word.flush()

        log_per_word.close()

        # plot_list = ["yellow:EN"]
        # for w in plot_list:
        #     plot_subgraph(en, w, 3, "en")
        #     plot_subgraph(biling, w, 3, "biling")

main()
