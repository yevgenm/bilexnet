import os
import re
from dutch import Graphnx
import json
import pickle
import numpy as np
import csv


def index_parser():
    fil = open("associationDataLemmas.csv", "r")
    indx = {}
    l = fil.readlines()[1:]
    for line in l:
        line = line.split(";")[2:]
        
        for word in line:
            word = word.strip('\n')
            word = word.strip('\t')
            word = word.strip('\r')
            word = word.strip('"')
            if word not in indx:
                #print(word)
                indx[word] = len(indx)
                
    return indx


def mat_constr():
    indx = index_parser()
    count = len(indx)
    M = np.zeros(count*count)
    M = M.reshape(count,count)
    M = M.astype(int)
    fil = open("associationDataLemmas.csv", "r")

    l = fil.readlines()[1:]
    for line in l:
        #print(line)
        line = line.split(";")[2:]
        
        
         
        for i in range(4):
            word = line[i]
            word = word.strip('\n')
            word = word.strip('\t')
            word = word.strip('\r')
            word = word.strip('"')
            
            if i == 0:
                row = indx[word]
                
            if i != 0:
                column = indx[word]
                
                M[row][column] = M[row][column] + 1
            
            
    return M


if __name__ == "__main__":
    M = mat_constr()
    print(M)
    #mult = np.dot(M,M)
    
    indx = index_parser()
    count = len(indx)
    P = np.zeros(count*count).reshape(count,count)
    P = P.astype(int)
    
    for r in range(count):
        for c in range(count):
            print(r,c)
            P[r][c] = np.dot(M[r][c],M[r][c])
            
            
