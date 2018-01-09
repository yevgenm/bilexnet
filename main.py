import os
import sys
from multiprocessing import Pool
from models import *
from parameters import *
import preprocessing as prep
import utils

def evaluate_model(args):
    test_cues, resp_lang, vertices_nl, vertices_en, norms_nl, norms_en, norms_bi, en_nl_dic, alignments, cognates, orth_sims_en, synt_coocs_en,\
    ups_max_base_an, ups_n_base_an, rhos_max_base_an, rhos_n_base_an, ups_max_base_sa, ups_n_base_sa, rhos_max_base_sa, rhos_n_base_sa, output_dir, \
    model_type, k_da, k_ea, k_te, k_cg, k_or, k_sy = args

    log_fn = "%s/log_%s_type_%s_da_%s_ea_%s_te_%s_cg_%s_or_%s_sy_%s" % (output_dir, resp_lang, model_type, str(k_da), str(k_ea), str(k_te), str(k_cg), str(k_or), str(k_sy))
    log_file = open(log_fn, "w")
    model_bi = LexNetBi(vertices_nl, vertices_en, norms_nl, norms_en, orth_sims_en, synt_coocs_en, en_nl_dic, cognates, alignments, k_da, k_ea, k_te, k_cg, k_or, k_sy, model_type)
    log_file.write("MODEL:%s-%s, K_DA: %i, K_EA: %i, K_TE:%i, K_CG:%i, K_OR: %i, K_SY:%i\n" % ("bi", model_type, k_da, k_ea, k_te, k_cg, k_or, k_sy))

    ups_max, ups_n, rhos_max, rhos_n = model_bi.test_network(test_cues, parameters["spreading depth"], resp_lang, norms_bi, True, log_file)
    log_file.close()

    return model_type, k_da, k_ea, k_te, k_cg, k_or, k_sy, ups_max, ups_n, rhos_max, rhos_n


def read_all_data():
    prep.activate_lemmatizers()
    norms_nl = prep.read_norm_data("./data/norms_preprocessed_nl.csv", "nl")
    norms_en = prep.read_norm_data("./data/norms_preprocessed_en.csv", "en")
    vertices_nl = list(set(list(norms_nl) + [resp for cue in norms_nl for resp in norms_nl[cue]]))
    vertices_en = list(set(list(norms_en) + [resp for cue in norms_en for resp in norms_en[cue]]))
    words_nl = [w[:-3] for w in vertices_nl]
    words_en = [w[:-3] for w in vertices_en]
    en_nl_dic = prep.read_dict("./data/en_nl_dictionary.csv", words_nl, words_en)
    alignments = prep.read_alignments("./data/word_alignments.csv", words_nl, words_en)
    cognates = prep.read_cognates("./data/cognates.csv", en_nl_dic, words_nl, words_en)
    norms_bi_nl = prep.read_bilingual_data("nl-nl")
    norms_bi_en = prep.read_bilingual_data("en-en")
    norms_bi = {"en-en": norms_bi_en, "nl-nl": norms_bi_nl}
    orth_sims_en = prep.read_orthography("./data/orth_sim_en.csv", words_en)
    synt_coocs_en = prep.read_cooccurences("./data/synt_coocc_en.csv", words_en)
    return norms_nl, norms_en, vertices_nl, vertices_en, en_nl_dic, alignments, cognates, norms_bi, orth_sims_en, synt_coocs_en


def run_group_comparisons(norms_nl, norms_en, norms_bi_nl, norms_bi_en, log_fn):

    with open(log_fn, 'w') as f:
        f.write("B1-1 & B1-2\tvs.\tB1-1 & P2\n")
        test_cues = sorted(list(set(norms_bi_en[2].keys()).intersection(set(norms_en.keys()))))
        ups_max_1, ups_n_1, rhos_max_1, rhos_n_1 = utils.compute_differences(norms_bi_en[0], norms_bi_en[1], test_cues)
        ups_max_2, ups_n_2, rhos_max_2, rhos_n_2 = utils.compute_differences(norms_bi_en[0], norms_bi_en[2], test_cues)
        utils.print_difference_stats(ups_max_1, ups_n_1, rhos_max_1, rhos_n_1, ups_max_2, ups_n_2, rhos_max_2, rhos_n_2, f)

        f.write("\nB1-1 & B2\tvs.\tB1-1 U B2 & M(EN)\n")
        ups_max_1, ups_n_1, rhos_max_1, rhos_n_1 = utils.compute_differences(norms_bi_en[0], norms_bi_en[2], test_cues)
        ups_max_2, ups_n_2, rhos_max_2, rhos_n_2 = utils.compute_differences(norms_bi_en["aggregated"], norms_en, test_cues)
        utils.print_difference_stats(ups_max_1, ups_n_1, rhos_max_1, rhos_n_1, ups_max_2, ups_n_2, rhos_max_2, rhos_n_2, f)

        f.write("\nB1-1 & B2\tvs.\tB1-1 U 2 & M(NL)\n")
        test_cues = sorted(list(set(norms_bi_nl[2].keys()).intersection(set(norms_nl.keys()))))
        test_cues = [w for w in test_cues if w not in ["voordeel", "motief"]]
        ups_max_1, ups_n_1, rhos_max_1, rhos_n_1 = utils.compute_differences(norms_bi_nl[0], norms_bi_nl[2], test_cues)
        ups_max_2, ups_n_2, rhos_max_2, rhos_n_2 = utils.compute_differences(norms_bi_nl["aggregated"], norms_nl, test_cues)
        utils.print_difference_stats(ups_max_1, ups_n_1, rhos_max_1, rhos_n_1, ups_max_2, ups_n_2, rhos_max_2, rhos_n_2, f)


def fit_models(vertices_nl, vertices_en, norms_nl, norms_en, norms_bi, en_nl_dic, alignments, cognates, orth_sims_en, synt_coocs_en, test_cues, task_direction):

    cue_lang, resp_lang = task_direction.split("-")

    model_mo = LexNetMo("ucs", norms_nl if resp_lang == "nl" else norms_en, resp_lang)

    output_dir = "./output/" + task_direction + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_base_file = open(output_dir + "log_base_an_" + task_direction, 'w')
    log_base_file.write("MODEL:%s\n" % ("base-an-" + resp_lang))
    ups_max_base_an, ups_n_base_an, rhos_max_base_an, rhos_n_base_an = model_mo.test_network(test_cues, 1, resp_lang, norms_bi, True, log_base_file)
    log_base_file.close()
    log_base_file = open(output_dir + "log_base_sa_" + task_direction, 'w')
    log_base_file.write("MODEL:%s\n" % ("base-sa-" + resp_lang))
    ups_max_base_sa, ups_n_base_sa, rhos_max_base_sa, rhos_n_base_sa = model_mo.test_network(test_cues, parameters["spreading depth"], resp_lang, norms_bi, True, log_base_file)
    log_base_file.close()

    log_per_word_fn = output_dir + "log_per_word_" + task_direction + ".tsv"

    if os.path.exists(log_per_word_fn):
        log_per_word = open(log_per_word_fn, 'a')
    else:
        log_per_word = open(log_per_word_fn, 'w')
        log_per_word.write("model_type\tk_da\tk_ea\tk_te\tk_cg\tk_or\tk_sy\t")
        for cue in test_cues:
            log_per_word.write("%s (u_max)\t%s (u_n)\t%s (r_max)\t%s (r_n)\t" % (cue, cue, cue, cue))
        log_per_word.write("\n")
        log_per_word.flush()

        log_per_word.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t" % ("base-an", "base-an", "base-an", "base-an", "base-an", "base-an", "base-an"))
        for idx in range(len(test_cues)):
            log_per_word.write("%.3f\t%.3f\t%.3f\t%.3f\t" % (ups_max_base_an[idx], ups_n_base_an[idx], rhos_max_base_an[idx], rhos_n_base_an[idx]))
        log_per_word.write("\n")
        log_per_word.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t" % ("base-sa", "base-sa", "base-sa", "base-sa", "base-sa", "base-sa", "base-sa"))
        for idx in range(len(test_cues)):
            log_per_word.write("%.3f\t%.3f\t%.3f\t%.3f\t" % (ups_max_base_sa[idx], ups_n_base_sa[idx], rhos_max_base_sa[idx], rhos_n_base_sa[idx]))
        log_per_word.write("\n")
        log_per_word.flush()

    meta_args = [test_cues, resp_lang, vertices_nl, vertices_en, norms_nl, norms_en, norms_bi, en_nl_dic, alignments, cognates, orth_sims_en, synt_coocs_en,
                 ups_max_base_an, ups_n_base_an, rhos_max_base_an, rhos_n_base_an, ups_max_base_sa, ups_n_base_sa, rhos_max_base_sa, rhos_n_base_sa, output_dir]

    params = [ [model_type, k_da, k_ea, k_te, k_cg, k_or, k_sy]
               for model_type in ["ucs", "cs"]
               for k_da in [0, 1, 5, 10, 15, 20, 25]
               for k_ea in [0, 1, 5, 10, 15, 20, 25]
               for k_te in [0, 1, 5, 10, 15, 20, 25]
               for k_cg in [0, 1, 5, 10, 15, 20, 25]
               for k_or in [0]
               for k_sy in [0]
               if (k_da + k_ea + k_te + k_cg > 0)
               ]

    args = [meta_args + param_set for param_set in params]

    pool = Pool(workers)
    for model_type, k_da, k_ea, k_te, k_cg, k_or, k_sy, ups_max, ups_n, rhos_max, rhos_n in pool.imap(evaluate_model, args):
        log_per_word.write("%s\t%d\t%d\t%d\t%d\t%d\t%d\t" % (model_type, k_da, k_ea, k_te, k_cg, k_or, k_sy))
        for idx in range(len(test_cues)):
            log_per_word.write("%.3f\t%.3f\t%.3f\t%.3f\t" % (ups_max[idx], ups_n[idx], rhos_max[idx], rhos_n[idx]))
        log_per_word.write("\n")
        log_per_word.flush()
    log_per_word.close()


if __name__ == "__main__":

    workers = 3
    if len(sys.argv) > 1:
        workers = int(sys.argv[1])

    norms_nl, norms_en, vertices_nl, vertices_en, en_nl_dic, alignments, cognates, norms_bi, orth_sims_en, synt_coocs_en = read_all_data()
    run_group_comparisons(norms_nl, norms_en, norms_bi["nl-nl"], norms_bi["en-en"], "./output/log_group_comparisons")

    task_direction = 'en-en'
    test_cues = sorted(list(set(norms_bi["en-en"][2].keys()).intersection(set(norms_en.keys()))))

    fit_models(vertices_nl, vertices_en, norms_nl, norms_en, norms_bi[task_direction]["aggregated"], en_nl_dic, alignments, cognates, orth_sims_en, synt_coocs_en, test_cues, task_direction)
