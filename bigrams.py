from google_ngram_downloader import readline_google_store
from collections import defaultdict
import string

word = "apple"
idx_right = ["_"]+list(string.ascii_lowercase)
idx_left = [word[0]]*len(idx_right)
indices = [i[0]+i[1] for i in zip(idx_left,idx_right)]


right_ctxt = defaultdict(int)
fname, url, records = next(readline_google_store(ngram_len=2, indices=indices))

try:
    record = next(records)
    w1, w2 = record.ngram.split(" ")

    while w1 != word:
        record = next(records)
        w1, w2 = record.ngram.split(" ")

    while record.ngram == word:
        record = next(records)
        w1, w2 = record.ngram.split(" ")
        right_ctxt[w2] += record.match_count

except StopIteration:
    pass

print(right_ctxt)