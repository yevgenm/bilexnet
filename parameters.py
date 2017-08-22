parameters = {
    "baseline depth": 3,
    "model depth": 3,
    #"frequency mode": "alignment",
    "frequency mode": "unigrams",
    "edge directory": "./biling_graph/unigram/",
    "orthographic threshold": 0.8,
    "cognate threshold": 0.5,
    "activation decay": 0.7,
    "jaccard k": 3,
    "apk k": 3,
    "retrieval algorithm": "spreading",
    "number of walks": 149,
    "frequency threshold": 1,
    "return allowed": False,
    "use syntagmatic edges": True,
    # "use frequencies": False,
    "use frequencies": True,
    # "orth edge type": "orth",
    "orth edge type": "cogn",
    "L1 to L2 reliance ratio": [0.75, 0.25],
}


extras = {
    "language mapping": {'E': ':EN', 'D': ':NL'},
    "noise list": ['x', '', None]
}
