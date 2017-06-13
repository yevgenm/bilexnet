from __future__ import print_function, unicode_literals #to make this work on Python 2 as well as Python 3

import csv
import frog

frog = frog.Frog(frog.FrogOptions(parser=False))

f_new = open("./Dutch/associationDataLemmas.csv", 'w')
writer = csv.writer(f_new, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)

lemmas = {}

with open("./Dutch/associationData.csv") as f:
    reader = csv.reader(f, delimiter=";")
    row1 = next(reader)
    writer.writerow(row1)
    for row in reader:
        for idx in range(3,6):
            token = row[idx].strip()
            if " " in token:
                row[idx] = "x"
            else:
                if token not in lemmas:
                    lemmas[token] = frog.process(row[idx])[0]['lemma']
                row[idx] = lemmas[token]
        writer.writerow(row)

f_new.close()

