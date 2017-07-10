import os
import re
from dutch import Graphnx
import json
import pickle
import numpy as np
import csv
import editdistance

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
                
if __name__ == "__main__":
    file_shrinker()
