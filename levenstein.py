import os
import re
from dutch import Graphnx
import json
import pickle
import numpy as np
import csv
import editdistance
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

def counter():
    fil = open("associationDataLemmas.csv", "r")
    dic = {}
    l = fil.readlines()[1:]
    for line in l:
        line = line.split(";")[2:]
        
        for word in line:
            word = word.strip('\n')
            word = word.strip('\t')
            word = word.strip('\r')
            word = word.strip('"')
            if word not in dic:
                #print(word)
                dic[word] = 1
                
            else:
                dic[word] = dic[word] + 1
                
    return dic

def file_shrinker():
    
    fil = open("associationDataLemmas.csv", "r")
    fil2 = open("shrunkDutch.csv","w")
    dic = counter()
    
    l = fil.readlines()[1:]
    for line in l:
        line = line.split(";")[2:]
        
        for word in line:
            word = word.strip('\n')
            word = word.strip('\t')
            word = word.strip('\r')
            word = word.strip('"')
            if dic[word] == 1:
                fil2.write( "x;")
            else:
                fil2.write( word + ";")
        fil2.write("\n")
    fil2.close()
            
                
             
def lcs(xstr, ystr):
    """
    >>> lcs('thisisatest', 'testing123testing')
    'tsitest'
    """
    if not xstr or not ystr:
        return ""
    x, xs, y, ys = xstr[0], xstr[1:], ystr[0], ystr[1:]
    if x == y:
        return x + lcs(xs, ys)
    else:
        return max(lcs(xstr, ys), lcs(xs, ystr), key=len)
    
    
    
def levdist():
    fil = open("shrunkDutch.csv", "r")
    dutch = set()
    l = fil.readlines()[1:]
    for line in l:
        line = line.split(";")[2:]
        
        for word in line:
            word = word.strip('\n')
            word = word.strip('\t')
            word = word.strip('\r')
            word = word.strip('"')
            dutch.add(word)
    print("dutch DONE!!!!!!!!!\n", len(dutch))
    efil = open("/Users/amirardalankalantaridehgahi/Desktop/school/stevensonRA/clone/bilexnet/modEAT.net","r")
    eng = set()
    l = efil.readlines()[33:23252]

    for line in l:
        line = line.split(" ",1)[-1]
        word = line.strip('\n')
        word = word.strip('\t')
        word = word.strip('\r')
        word = word.strip('"')
        eng.add(word)


    print("eng DONE!!!!!!!!!\n", len(eng))
    lev = []
    malemad = []
    c = 0
    for engw in eng:
        c = c+ 1
        if c%500 == 0:
            print(c)
        for dutchw in dutch:
            
            lev.append((engw,dutchw, editdistance.eval(engw,dutchw)))
            #malemad.append((engw,dutchw, lcs(engw,dutchw)))
    return lev , malemad 
                
                
def southlev():
    FILES = ["Cue_Target_Pairs.txt", "Cue_Target_Pairs1.txt", "Cue_Target_Pairs2.txt", "Cue_Target_Pairs3.txt", "Cue_Target_Pairs4.txt" , "Cue_Target_Pairs5.txt", "Cue_Target_Pairs6.txt","Cue_Target_Pairs7.txt"]

    DICT = dict()
    with open('southflor.lev.csv', 'w') as handle:
        
        
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
                
                
                levi = 1 - ((editdistance.eval(target,cue))/ max(len(cue),len(target)))
                print(levi,target, cue)
                if levi>=0.5:
                    handle.write(target + ',' + cue + ',' + str(levi)+'\n')
                
                
        print(count)

    
    
    
    
def cognates():
    fil = open('dict/dictionary.csv')
    l = fil.readlines()
    l = l[1:]
    dic = {}
    for i in l:
        line = i.strip('\n')
        line = line.strip('\t')
        line = line.split(',')
        
        if line[0] not in dic:
            dic[line[0]] = []
            dic[line[0]].append(line[1].strip())
            #print(dic,'ssssssssssssssssssssssss')
            
        else:
            #print(dic,'ssssssssssssssssssssssss')
            dic[line[0]].append(line[1].strip())
                              
        
    
            
    fil = open('levdist.csv')
    l = fil.readlines()
    l = l[1:]
    
    with open('cognates.csv', 'w') as cogs:
        for line in l:
            ls = line.strip('\n').split(',')
            if float(ls[2]) >= 0.7:
                if  ls[0] in dic and  ls[1] in dic[ls[0]]:
                    cogs.write(ls[0] + ',' + ls[1] + ',' + ls[2]+ '\n')
                    
    
    
    
    
    
    
    
    
    
    
    
    

if __name__ == "__main__":
    cognates()
    
