from networks_igraph import *
from multiprocessing import Pool
from parameters import *

def test_model(args):

    mode, test_list, test_condition, fn_nl, fn_en, en_nl_dic, tvd_base_an, rbd_base_an, jac_base_an, apk_base_an, apk_base_10_an, tvd_base_sa, rbd_base_sa, jac_base_sa, apk_base_sa, apk_base_10_sa, gold_dict, res_dir_cond, L1_assoc_coeff, L2_assoc_coeff, TE_coeff, orth_coeff, asymm_ratio = args
    log_fn = res_dir_cond + "log_" + test_condition + "_L1_assoc_" + str(L1_assoc_coeff) + "_L2_assoc_" + str(
        L2_assoc_coeff) + "_TE_" + str(TE_coeff) + "_orth_" + str(orth_coeff) + "_asymm_" + str(asymm_ratio)
    log_file = open(log_fn, "w")
    biling = LexNetBi(fn_nl, fn_en, en_nl_dic, L1_assoc_coeff, L2_assoc_coeff, TE_coeff, orth_coeff, asymm_ratio, mode)

    if test_condition == "ED":
        dict_for_filtering = en_nl_dic
    elif test_condition == "DE":
        dict_for_filtering = utils.invert_dict(en_nl_dic)
    else:
        dict_for_filtering = None

    log_file.write("NET:%s, DEPTH:%i, L1 ASSOC: %i, L2 ASSOC: %i, TE:%i, ORTH:%i, ASYMM: %i\n" % ("bi", parameters[
        "model depth"], L1_assoc_coeff, L2_assoc_coeff, TE_coeff, orth_coeff, asymm_ratio))
    tvd_m, rbd_m, jac_m, apk_m, apk_m_10 = biling.test_network(test_list, parameters["model depth"], test_condition, translation_dict=dict_for_filtering, gold_full=gold_dict, verbose=True, log_file=log_file)

    log_file.write("TVD t-test vs. AN: T=%.2f, p=%.3f\n" % (ttest_rel(tvd_base_an, tvd_m)[0], ttest_rel(tvd_base_an, tvd_m)[1]))
    log_file.write("RBD t-test vs. AN: T=%.2f, p=%.3f\n" % (ttest_rel(rbd_base_an, rbd_m)[0], ttest_rel(rbd_base_an, rbd_m)[1]))
    log_file.write("JAC t-test vs. AN: T=%.2f, p=%.3f\n" % (ttest_rel(jac_base_an, jac_m)[0], ttest_rel(jac_base_an, jac_m)[1]))
    log_file.write("APK(k) t-test vs. AN: T=%.2f, p=%.3f\n" % (ttest_rel(apk_base_an, apk_m)[0], ttest_rel(apk_base_an, apk_m)[1]))
    log_file.write("APK(10) t-test vs. AN: T=%.2f, p=%.3f\n" % (ttest_rel(apk_base_10_an, apk_m_10)[0], ttest_rel(apk_base_10_an, apk_m_10)[1]))

    log_file.write("TVD t-test vs. SA: T=%.2f, p=%.3f\n" % (ttest_rel(tvd_base_sa, tvd_m)[0], ttest_rel(tvd_base_sa, tvd_m)[1]))
    log_file.write("RBD t-test vs. SA: T=%.2f, p=%.3f\n" % (ttest_rel(rbd_base_sa, rbd_m)[0], ttest_rel(rbd_base_sa, rbd_m)[1]))
    log_file.write("JAC t-test vs. SA: T=%.2f, p=%.3f\n" % (ttest_rel(jac_base_sa, jac_m)[0], ttest_rel(jac_base_sa, jac_m)[1]))
    log_file.write("APK(k) t-test vs. SA: T=%.2f, p=%.3f\n" % (ttest_rel(apk_base_sa, apk_m)[0], ttest_rel(apk_base_sa, apk_m)[1]))
    log_file.write("APK(10) t-test vs. SA: T=%.2f, p=%.3f\n" % (ttest_rel(apk_base_10_sa, apk_m_10)[0], ttest_rel(apk_base_10_sa, apk_m_10)[1]))

    log_file.close()

    return(L1_assoc_coeff, L2_assoc_coeff, TE_coeff, orth_coeff, asymm_ratio, tvd_m, rbd_m, jac_m, apk_m, apk_m_10)


def main():

    res_dir = sys.argv[1]
    if res_dir[-1] != "/":
        res_dir += "/"
    workers = int(sys.argv[2])

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    fn_en = "./SF_norms/sothflorida_complete.csv"
    fn_nl = "./Dutch/dutch_final.csv"

    en_nl_dic = utils.read_dict("./dict/dictionary.csv")

    monoling = {"E": LexNetMo(fn=fn_en, language="en"),
                "D": LexNetMo(fn=fn_nl, language="nl")}

    gold_dict = read_test_data()

    test_wordlist = {"E": utils.filter_test_list(monoling["E"].G, sorted(gold_dict['EE'].keys())),
                     "D": utils.filter_test_list(monoling["D"].G, sorted(gold_dict['DD'].keys()))}

    # test = ['ED','DE']
    test = ['EE']
    mode = parameters["orth edge type"]

    for test_condition in test:

        res_dir_cond = res_dir + test_condition + "/"
        if os.path.exists(res_dir_cond):
            pass
            #if os.listdir(res_dir_cond) != []:
            #    sys.exit("Result directory not empty!")
        else:
            os.makedirs(res_dir_cond)

        cue_lang = test_condition[0]
        target_lang = test_condition[1]

        test_list = test_wordlist[cue_lang]

        if cue_lang==target_lang:
            log_base_file = open(res_dir_cond + "log_base_sa_"+test_condition, 'w')
            log_base_file.write("NET:%s, DEPTH:%i\n" % (target_lang, parameters["baseline depth"]))
            tvd_base_sa, rbd_base_sa, jac_base_sa, apk_base_sa, apk_base_10_sa = monoling[target_lang].test_network(test_list, parameters["baseline depth"], test_condition, gold_full=gold_dict, verbose=True, log_file=log_base_file)
            log_base_file.close()
            log_base_file = open(res_dir_cond + "log_base_an_"+test_condition, 'w')
            log_base_file.write("NET:%s, DEPTH:%i\n" % (target_lang, 1))
            tvd_base_an, rbd_base_an, jac_base_an, apk_base_an, apk_base_10_an = monoling[target_lang].test_network(test_list, 1, test_condition, gold_full=gold_dict, verbose=True, log_file=log_base_file)
            log_base_file.close()
        else:
            tvd_base_sa = rbd_base_sa = jac_base_sa = apk_base_sa = apk_base_10_sa = [0] * len(test_list)
            tvd_base_an = rbd_base_an = jac_base_an = apk_base_an = apk_base_10_an = [0] * len(test_list)

        log_per_word_fn = res_dir_cond + "log_per_word_"+test_condition+".tsv"
        #if os.path.exists(log_per_word_fn): append_write = 'a'
        #else: append_write = 'w'  # make a new file if not
        log_per_word = open(log_per_word_fn, 'w')
        log_per_word.write("L1_assoc\tL2_assoc\tTE\torth\tasymm\t")
        for w in test_list:
            log_per_word.write("%s (tvd)\t%s (rbd)\t%s (jac)\t%s (apk_k)\t%s (apk_10)\t" % (w, w, w, w, w))

        log_per_word.write("\n")
        log_per_word.flush()

        log_per_word.write("%s\t%s\t%s\t%s\t%s\t" % ("base-an", "base-an", "base-an", "base-sa", "base-an"))
        for idx in range(len(test_list)):
            log_per_word.write("%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t" % (tvd_base_an[idx], rbd_base_an[idx], jac_base_an[idx], apk_base_an[idx], apk_base_10_an[idx]))
        log_per_word.write("\n")
        log_per_word.write("%s\t%s\t%s\t%s\t%s\t" % ("base-sa", "base-sa", "base-sa", "base-sa", "base-sa"))
        for idx in range(len(test_list)):
            log_per_word.write("%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t" % (tvd_base_sa[idx], rbd_base_sa[idx], jac_base_sa[idx], apk_base_sa[idx], apk_base_10_sa[idx]))
        log_per_word.write("\n")
        log_per_word.flush()

        meta_args = [mode, test_list, test_condition, fn_nl, fn_en, en_nl_dic, tvd_base_an, rbd_base_an, jac_base_an, apk_base_an, apk_base_10_an, tvd_base_sa, rbd_base_sa, jac_base_sa, apk_base_sa, apk_base_10_sa, gold_dict, res_dir_cond]

        par = [ [L1_assoc_coeff, L2_assoc_coeff, TE_coeff, orth_coeff, asymm_ratio]
                for L1_assoc_coeff in [1, 2, 5, 10, 20]
                #for L2_assoc_coeff in [0, 1, 2, 5, 10, 20]
                for L2_assoc_coeff in [1, 2, 5, 10, 20]
                for TE_coeff in [1, 2, 5, 10, 20]
                #for orth_coeff in [0, 1, 2, 5, 10, 20]
                for orth_coeff in [1, 2, 5, 10, 20]
                for asymm_ratio in [1]
                # if L1_assoc_coeff==1 or not (L1_assoc_coeff==TE_coeff==L2_assoc_coeff==orth_coeff) and L1_assoc_coeff >= L2_assoc_coeff
                # for (L1_assoc_coeff, L2_assoc_coeff, TE_coeff, orth_coeff, asymm_ratio) in [[5,2,2,0,1],[20,5,5,0,1],[5,1,1,0,1],[2,2,1,0,1]]
                # for (L1_assoc_coeff, L2_assoc_coeff, TE_coeff, orth_coeff, asymm_ratio) in [[5,10,5,5,1]]
                ]

        args = [meta_args + par_set for par_set in par]

        #for a in args:
        #    run_test(a)

        pool = Pool(workers)
        for L1_assoc_coeff, L2_assoc_coeff, TE_coeff, orth_coeff, asymm_ratio, tvd_m, rbd_m, jac_m, apk_m, apk_m_10 in pool.imap(test_model, args):
            log_per_word.write("%d\t%d\t%d\t%d\t%d\t" % (L1_assoc_coeff, L2_assoc_coeff, TE_coeff, orth_coeff, asymm_ratio))
            for idx in range(len(test_list)):
                log_per_word.write("%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t" % (tvd_m[idx], rbd_m[idx], jac_m[idx], apk_m[idx], apk_m_10[idx]))
            log_per_word.write("\n")
            log_per_word.flush()

        log_per_word.close()

        # plot_list = ["yellow:EN"]
        # for w in plot_list:
        #     plot_subgraph(en, w, 3, "en")
        #     plot_subgraph(biling, w, 3, "biling")

main()
