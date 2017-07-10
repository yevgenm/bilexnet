import csv
import frog
from nltk import pos_tag
from nltk import WordNetLemmatizer
from nltk.corpus import wordnet as wn


frog = frog.Frog(frog.FrogOptions(parser=False))
lemmatizer = WordNetLemmatizer()

def wn_lemmatizer(word):
    tag = pos_tag([word])[0][1]    # Converting it to WordNet format.
    mapping = {'N': wn.NOUN, 'V': wn.VERB, 'R': wn.ADV,'J': wn.ADJ}
    tag_wn = mapping.get(tag[0], wn.NOUN)
    lemma = lemmatizer.lemmatize(word, tag_wn)
    return lemma.encode('ascii', 'ignore').decode('ascii')

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

def read_test_file(fn):
    # An auxiliary function that reads a single test file.

    conditions = fn.split("/")[-1].split('.')[0].split('-')
    target_langs = [c[1] for c in conditions]
    n_conds = len(conditions)
    with open(fn) as f, open(fn[:-4]+".lemmas.csv", 'w') as fw:
        test_reader = csv.reader(f, delimiter=",")
        test_writer = csv.writer(fw, delimiter=",")
        row1 = next(test_reader)
        test_writer.writerow(row1)
        for row in test_reader:
            new_row = [row[0], row[1], row[2]]
            for idx in range(3, len(row)):
                response = preprocess_word(row[idx])
                if response == "":
                    lemma = ""
                else:
                    cond_idx = idx % n_conds
                    target_lang = target_langs[cond_idx]
                    lemma = lemmatize_word(response, target_lang)
                new_row.append(lemma)
            test_writer.writerow(new_row)

def main():
    read_test_file("./vanhell/DD1-DE2-DD3.csv")
    read_test_file("./vanhell/DE1-DD2.csv")
    read_test_file("./vanhell/ED1-EE2.csv")
    read_test_file("./vanhell/EE1-ED2-EE3.csv")

main()