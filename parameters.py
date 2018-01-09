from nltk.corpus import wordnet as wn

parameters = {
    "cognate threshold": 0.5,
    "cut-off threshold": 0.01,
    "edge directory": "./output/edge_files",
    "evaluation n value": 3,
    "frequency threshold": 1,
    "spreading depth": 3,
    "orthographic threshold": 0.75,
}


constants = {
    "language mapping": {'E': ':en', 'D': ':nl'},
    "noise list": ['x', '', None],
    "wn mapping": {'N': wn.NOUN, 'V': wn.VERB, 'R': wn.ADV, 'J': wn.ADJ}
}
