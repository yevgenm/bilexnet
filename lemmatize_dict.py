import os
import csv
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
    return lemma.encode('ascii', 'ignore').decode('ascii')

def lemmatize_word(word, lang):
    if lang == "D":
        word = frog.process(word)[0]['lemma']
    else:
        word = wn_lemmatizer(word)
    return word

def lemmatize_dict(fn):
    # An auxiliary function that reads a single test file.

    with open(fn) as f, open(fn[:-4]+".lemmas.csv", 'w') as fw:
        test_reader = csv.reader(f, delimiter=",")
        test_writer = csv.writer(fw, delimiter=",")
        row1 = next(test_reader)
        test_writer.writerow(row1)
        for row in test_reader:
            en = row[0]
            nl = row[1]
            en_lemma = lemmatize_word(en, "E")
            nl_lemma = lemmatize_word(nl, "D")
            test_writer.writerow([en_lemma,nl_lemma])

def main():

    lemmatize_dict("./dict/dictionary.csv")


main()
