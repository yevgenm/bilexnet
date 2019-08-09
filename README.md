To run the model, you need to obtain bilingual word association data collected by van Hell and de Groot (1998), which are not available online:
- obtain the Excel files from Janet van Hell
- convert them to .csv and save the 4 files (DD1-DE2-DD3.csv, DE1-DD2.csv, ED1-EE2.csv, and EE1-ED2-EE3.csv) in ./data/bilingual/
- run preprocess_bilingual_data.py

The other data sources are available online:

1. COCA non-case-sensitive bigram and trigram frequency lists:
https://www.ngrams.info/download_coca.asp
Place w2_.txt and w3_.txt in ./data/coca/

2. Dictionaries:
- eng-nld.tei from FreeDict:
https://github.com/freedict/fd-dictionaries/tree/master/eng-nld:
- nld-eng.tei from FreeDict:
https://github.com/freedict/fd-dictionaries/tree/master/nld-eng
- EN>NL from dict.cc:
https://www1.dict.cc/translation_file_request.php
Place the 3 files (eng-nld.tei, nld-eng.tei, and dict.cc) in ./data/dict/

3. University of South Florida association norms:
http://w3.usf.edu/FreeAssociation/AppendixA/index.html
Place 8 files (Cue_Target_Pairs.A-B, ..., Cue_Target_Pairs.T-Z) in ./data/norms_en

4. Dutch word association data from Small World of Words:
https://smallworldofwords.org/en/project/research
Place associationData.csv in ./data/norms_nl

After obtaining all data, run main.py, preferably with Frog installed:
http://languagemachines.github.io/frog/
