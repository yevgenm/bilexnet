parameters = {
    "baseline depth": 3,
    "model depth": 3,
    "orthographic threshold": 0.8,
    "cognate threshold": 0.5,
    "activation decay": 0.7,
    "jaccard k": 3,
    "apk k": 3,
    "retrieval algorithm": "walks",
    "number of walks": 100,
    "frequency threshold": 1,
    "return allowed": True,
    "use syntagmatic edges": True,
    # "use frequencies": False,
    "use frequencies": True,
    # "orth edge type": "orth",
    "orth edge type": "cogn"
}


extras = {
    "language mapping": {'E': ':EN', 'D': ':NL'},
    "noise list": ['x', '', None]
}
