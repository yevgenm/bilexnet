import os
import frog
from nltk import pos_tag
from nltk import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import csv

frog = frog.Frog(frog.FrogOptions(parser=False))
lemmatizer = WordNetLemmatizer()

def wn_lemmatizer(word):
    tag = pos_tag([word])[0][1]    # Converting it to WordNet format.
    mapping = {'N': wn.NOUN, 'V': wn.VERB, 'R': wn.ADV,'J': wn.ADJ}
    tag_wn = mapping.get(tag[0], wn.NOUN)
    lemma = lemmatizer.lemmatize(word, tag_wn)
    return lemma

def preprocess_word(w):
    # An auxiliary function to clean test files.
    if "f-" in w:
        w = w[2:]
    if w[:3] == "vk-":
        w = w[3:]
    if w[:2] == "vk":
        w = w[2:]
    if w=="geen":
        return ""
    if "(" in w:
        w = w[:w.index("(")]
    if w[:3] == "to-":
        w = w[3:]
    if w[:3] == "de-":
        w = w[3:]
    if w[:4] == "een-":
        w = w[4:]
    if w[:4] == "the-":
        w = w[4:]
    if "-" in w:
        return ""
    return w

def lemmatize_word(word, lang):
    if lang == "D":
        word = frog.process(word)[0]['lemma']
    else:
        word = wn_lemmatizer(word)
    return word

def read_test_file(fn, spell_dict, log_fn):

    if os.path.exists(log_fn):
        append_write = 'a'
    else:
        append_write = 'w'
    log = open(log_fn, append_write)

    conditions = fn.split("/")[-1].split('.')[0].split('-')
    target_langs = [c[1] for c in conditions]
    n_conds = len(conditions)
    cue_lang = conditions[0][0]
    with open(fn) as f, open(fn[:-4]+".lemmas.csv", 'w') as fw:
        test_reader = csv.reader(f, delimiter=",")
        test_writer = csv.writer(fw, delimiter=",")
        row1 = next(test_reader)
        test_writer.writerow(row1)
        for row in test_reader:
            cue = row[1]
            cue_lemma = lemmatize_word(cue, cue_lang)
            if cue_lemma in spell_dict and cue_lang == "E":
                cue_lemma = spell_dict[cue_lemma]
            log.write(cue_lang+"\t"+cue+"\t"+cue_lemma+"\n")
            new_row = [row[0], cue_lemma, row[2]]
            for idx in range(3, len(row)):
                response = preprocess_word(row[idx])
                if response == "":
                    lemma = "x"
                else:
                    true_idx = idx - 3
                    cond_idx = true_idx % n_conds
                    target_lang = target_langs[cond_idx]
                    lemma = lemmatize_word(response, target_lang).strip()
                    if lemma in spell_dict and target_lang=="E":
                        lemma = spell_dict[lemma]
                    log.write(target_lang + "\t" + response + "\t" + lemma + "\n")
                new_row.append(lemma)
            test_writer.writerow(new_row)

    log.close()

def main():
    with open('./spelling_correction', 'r') as spelling:
        reader = csv.reader(spelling, delimiter="\t")
        spell_dict = {rows[0]: rows[1] for rows in reader}

    log_fn = "./log_lemmas"
    read_test_file("./EE1-ED2-EE3.csv", spell_dict, log_fn)
    read_test_file("./DD1-DE2-DD3.csv", spell_dict, log_fn)
    read_test_file("./DE1-DD2.csv", spell_dict, log_fn)
    read_test_file("./ED1-EE2.csv", spell_dict, log_fn)


main()
