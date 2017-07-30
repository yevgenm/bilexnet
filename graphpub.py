from collections import Counter
import copy
import csv
from igraph import *
import itertools
import ml_metrics as metrics
import numpy as np
import os
import utils
from pajek_converter import convert_pajek
import pandas
from parameters import extras, parameters
import random
from scipy.stats import ttest_rel
from test_data_reader import read_test_data
from numpy.random import choice


fn = 'graph/samp1'
g = biling = read(fn, format="ncol")

clustering_coefficeint = g.transitivity_undirected()  #Calculates the local transitivity (clustering coefficient) of the given vertices in the graph.
short_paths = g.shortest_paths()  #returns a matrix #pass in the edge weights or else all the edges are assumed to have uniform weight 
                                    # we can take the average ourselves perhaps
                                    
matrix =  np.array(short_paths)
row_average = np.mean(matrix, axis=1)
final_average = np.mean(row_average, axis=0)  #this should be the final average for short paths lenght


cluster = g.community_optimal_modularity() # 
modularity_coefficient = cluster.modularity
