from igraph import *
import csv
import os
from collections import Counter
import utils
import pandas
import ml_metrics as metrics
import random
from scipy.stats import ttest_rel
import numpy as np
from pajek_converter import convert_pajek
import pickle
import itertools
import copy
import pickle
import sys
from os import listdir
from os.path import isfile, join


FILES = ["Cue_Target_Pairs.txt", "Cue_Target_Pairs1.txt", "Cue_Target_Pairs2.txt", "Cue_Target_Pairs3.txt", "Cue_Target_Pairs4.txt" , "Cue_Target_Pairs5.txt", "Cue_Target_Pairs6.txt","Cue_Target_Pairs7.txt"]

dir = "./SF_norms/"
if not os._exists(FILES[0]):
    FILES = [dir+f for f in listdir(dir) if isfile(join(dir, f))]


DICT = dict()
with open('southflor.pickle', 'wb') as handle:
    
    
    count = 0
    for fil in FILES:
        print(fil)
        f = open(fil, "r",encoding="latin-1")
        l = f.readlines()
        l = l[4:-3]
        print(len(l))
        for line in l:
            line = line
            line = line.split(",")
            cue = line[0].lower().strip()
            target = line[1].lower().strip()
            average = float(line[5])
            
            #print(cue)
            count = count+1
            if cue not in DICT:
                DICT[cue] = dict()
                
            if target not in DICT[cue]:
                DICT[cue][target] = average
                
    print(count)
    pickle.dump(DICT, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
