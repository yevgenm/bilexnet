import os
import re
from dutch import Graphnx
import json
import pickle
import numpy as np
import csv

def nld_eng_parser():
    '''
    returns a dictionary of Dutch-->English
    
    every Dutch word has a corresponding list of English words that it is translated to.
    
    The direct variable is the location of the .tei file
    
    '''
    
    direct = '/Users/amirardalankalantaridehgahi/Desktop/school/stevensonRA/dict'  #change it
    fil = open(os.path.join(direct, "nld-eng.tei"), "r")

    tei = fil.readlines()

    DICT = {}

    new_entry = False
    entrant = None
    count = 0
    for line in tei:
        if line.strip() == "<entry>":
            new_entry = True
            count = count + 1
        
        if line.strip() =="</entry>":
            new_entry = False
            entrant = None
            
        if new_entry == True:
            
            if line.strip()[:6] == "<orth>":
                line = re.sub(r'<orth>', '',line.strip()) 
                line = re.sub(r'</orth>', '',line)
                if line not in DICT:
                    DICT[line] = []
                entrant = line
                print(entrant,'  entry')
                
            if line.strip()[:7] == "<quote>":
                line = re.sub(r'<quote>', '',line.strip()) 
                line = re.sub(r'</quote>', '',line)
                
                DICT[entrant].append(line)
                print(line,'  trans')
        
        #if count == 20:
        #    break
    print(count, '  count')
    return DICT

def eng_nld_parser():
    '''
    returns a dictionary of English-->Dutch
    
    every English word has a corresponding list of Dutch words that it is translated to.
    
    The direct variable is the location of the .tei file
    
    '''
    
    direct = '/Users/amirardalankalantaridehgahi/Desktop/school/stevensonRA/dict'  #change it
    fil = open(os.path.join(direct, "eng-nld.tei"), "r")

    tei = fil.readlines()

    DICT = {}

    new_entry = False
    entrant = None
    count = 0
    for line in tei:
        if line.strip() == "<entry>":
            new_entry = True
            count = count + 1
        
        if line.strip() =="</entry>":
            new_entry = False
            entrant = None
            
        if new_entry == True:
            
            if line.strip()[:6] == "<orth>":
                line = re.sub(r'<orth>', '',line.strip()) 
                line = re.sub(r'</orth>', '',line)
                if line not in DICT:
                    DICT[line] = []
                entrant = line
                print(entrant,'  entry')
                
            if line.strip()[:7] == "<quote>":
                line = re.sub(r'<quote>', '',line.strip()) 
                line = re.sub(r'</quote>', '',line)
                
                DICT[entrant].append(line)
                print(line,'  trans')
        
        #if count == 20:
        #    break
    print(count, '  count')
    return DICT



def cc_parser():
    dikt = {}
    f = open("dict_cc_en_nl.tsv")
    l = f.readlines()
    for i in l:
        line = i.split('\t')
        if line[0] not in dikt:
            dikt[line[0]] = []
        dikt[line[0]].append(line[1].strip('\n'))
    return dikt



def nld_eng_reverser():
    dikt = nld_eng_parser()
    DIKT = {}
    for nld in dikt:
        for eng in dikt[nld]:
            if eng not in DIKT:
                DIKT[eng] = set()
            DIKT[eng].add(nld)
            
    return DIKT
            
        
    
    
    
def parser_helper(final, DICT):
    for eng in DICT:
        for nld in DICT[eng]:
            if eng not in final:
                final[eng] = set()
            final[eng].add(nld)
            
    return final
    
    
def full_dict():
    cc = cc_parser()
    rev = nld_eng_reverser()
    eng = eng_nld_parser()
    
    final = {}
    raw_input()
    final = parser_helper(final,cc)
    raw_input()
    final = parser_helper(final,rev)
    raw_input()
    final = parser_helper(final, eng)
    raw_input()
    return final
    
    
    

def csvwriter():
    d = full_dict()
    with open('dictionary.csv', 'w') as csvfile:
        fieldnames = ['English', 'Dutch']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for eng in d:
            for dutch in d[eng]:
                if (  len(eng.split(' '))  <2 ) and ( len(dutch.split(' ')) <2 ):
                    writer.writerow({'English': eng, 'Dutch': dutch})
                    print(eng,dutch)
        
        
        
if __name__ == "__main__":
    
    '''
    This script shows how many of the words in the lexical Dutch graph have a corresponding English tranlastion.
    
    '''
    '''
    dikt = nld_eng_parser()
    raw_input()
    G = Graphnx()
    nodes = G.nodes()
    
    hits = 0
    miss = 0
    
    for n in nodes:
        if n in dikt:
            hits = hits + 1
            
        else:
            print(n, '  MISSSEEEEDDDDDDDD')
            miss = miss + 1
            
    print(hits , '  hits')
    print(miss , '   misses')
    '''
    
    
    
    '''
    94 entries have no tranlastion
    '''
    '''
    dikt = eng_nld_parser()
    
    dutch_domain = []
    for i in dikt.values():
        for j in i:
            dutch_domain.append(j)
            
    
    raw_input()
    G = Graphnx()
    nodes = G.nodes()
    
    hits = 0
    miss = 0
    
    for n in nodes:
        if n in dutch_domain:
            hits = hits + 1
            
        else:
            print(n, '  MISSSEEEEDDDDDDDD')
            miss = miss + 1
            
    print(hits , '  hits')
    print(miss , '   misses')
    '''
    
    csvwriter()
    
    

