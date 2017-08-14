from google_ngram_downloader import readline_google_store
from collections import defaultdict
import igraph

def get_words(record):
    w1, w2 = record.ngram.lower().split(" ")
    w1 = w1.split("_")[0]
    w2 = w2.split("_")[0]
    return w1, w2

G = igraph.read("./SF_norms/sothflorida_complete.csv_dump", format="ncol")
words = [w[:-3] for w in G.vs['name']]

words = words[:3]

word_dict = {}
for w in words:
    idx = w[:2]
    if idx not in word_dict:
        word_dict[idx] = set()
    word_dict[idx].add(w)

indices = list(word_dict.keys())

right_ctxt = dict()

for idx in indices:
    data = readline_google_store(ngram_len=2, indices=[idx])
    fname, url, records = next(data)

    tuples = [(get_words(r), r.match_count) for r in records]


    try:
        record = next(records)
        #w1, w2 = get_words(record)
        w1, w2 = record.ngram.lower().split(" ")
        w1 = w1.split("_")[0]
        w2 = w2.split("_")[0]

        while record:

            if w1 in word_dict[idx] and w2 in words:
                if w1 not in right_ctxt:
                    right_ctxt[w1] = dict()
                if w2 not in right_ctxt[w1]:
                    right_ctxt[w1][w2] = record.match_count
                else:
                    right_ctxt[w1][w2] += record.match_count

            record = next(records)
            #w1, w2 = get_words(record)
            w1, w2 = record.ngram.lower().split(" ")
            w1 = w1.split("_")[0]
            w2 = w2.split("_")[0]


    except StopIteration:
        pass

print(right_ctxt)



# idx_right = ["_"]+list(string.ascii_lowercase)
# idx_left = [word[0]]*len(idx_right)
# indices = [i[0]+i[1] for i in zip(idx_left,idx_right)]
