####!/usr/bin/env python

"""
Functions for data IO for neural network training.
"""

from __future__ import print_function # Might not be necessary, used to import print function in older python version
import argparse
import sys
import os
import time
import matplotlib.pyplot as plt
from operator import add
import math
import numpy as np
import yaml

def mkdir(outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

# Function to load yaml configuration file
def load_config(config_name):
    with open(config_name) as file:
        config = yaml.safe_load(file)

    return config

def enc_list_bl_max_len(aa_seqs, blosum, max_seq_len, padding = "right"):
    '''
    blosum encoding of a list of amino acid sequences with padding 
    to a max length

    parameters:
        - aa_seqs : list with AA sequences
        - blosum : dictionnary: key= AA, value= blosum encoding
        - max_seq_len: common length for padding
    returns:
        - enc_aa_seq : list of np.ndarrays containing padded, encoded amino acid sequences
    '''

    # encode sequences:
    sequences=[]
    blosum = get_encoding_scheme(name = blosum)
    for seq in aa_seqs:
        e_seq= np.zeros((len(seq),len(blosum["A"])))
        count=0
        for aa in seq:
            if aa in blosum:
                e_seq[count]=blosum[aa]
                count+=1
            else:
                sys.stderr.write("Unknown amino acid in peptides: "+ aa +", encoding aborted!\n")
                sys.exit(2)

        sequences.append(e_seq)

    # pad sequences:
    #max_seq_len = max([len(x) for x in aa_seqs])
    n_seqs = len(aa_seqs)
    n_features = sequences[0].shape[1]

    enc_aa_seq = -5*np.ones((n_seqs, max_seq_len, n_features))
    if padding == "right":
        for i in range(0,n_seqs):
            enc_aa_seq[i, :sequences[i].shape[0], :n_features] = sequences[i]
            
    elif padding == "left":
        for i in range(0,n_seqs):
            enc_aa_seq[i, max_seq_len-sequences[i].shape[0]:max_seq_len, :n_features] = sequences[i]
    
    else:
        print("Error: No valid padding has been chosen.\nValid options: 'right', 'left'")
        

    return enc_aa_seq



##############################
# Different encoding schemes #
##############################

def get_encoding_scheme(name = 'blosum50_20aa'):
    if name == 'blosum50_20aa':
        scheme = {'A': np.array([ 5,-2,-1,-2,-1,-1,-1, 0,-2,-1,-2,-1,-1,-3,-1, 1, 0,-3,-2, 0]),
                  'R': np.array([-2, 7,-1,-2,-4, 1, 0,-3, 0,-4,-3, 3,-2,-3,-3,-1,-1,-3,-1,-3]),
                  'N': np.array([-1,-1, 7, 2,-2, 0, 0, 0, 1,-3,-4, 0,-2,-4,-2, 1, 0,-4,-2,-3]),
                  'D': np.array([-2,-2, 2, 8,-4, 0, 2,-1,-1,-4,-4,-1,-4,-5,-1, 0,-1,-5,-3,-4]),
                  'C': np.array([-1,-4,-2,-4,13,-3,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-5,-3,-1]),
                  'Q': np.array([-1, 1, 0, 0,-3, 7, 2,-2, 1,-3,-2, 2, 0,-4,-1, 0,-1,-1,-1,-3]),
                  'E': np.array([-1, 0, 0, 2,-3, 2, 6,-3, 0,-4,-3, 1,-2,-3,-1,-1,-1,-3,-2,-3]),
                  'G': np.array([ 0,-3, 0,-1,-3,-2,-3, 8,-2,-4,-4,-2,-3,-4,-2, 0,-2,-3,-3,-4]),
                  'H': np.array([-2, 0, 1,-1,-3, 1, 0,-2,10,-4,-3, 0,-1,-1,-2,-1,-2,-3, 2,-4]),
                  'I': np.array([-1,-4,-3,-4,-2,-3,-4,-4,-4, 5, 2,-3, 2, 0,-3,-3,-1,-3,-1, 4]),
                  'L': np.array([-2,-3,-4,-4,-2,-2,-3,-4,-3, 2, 5,-3, 3, 1,-4,-3,-1,-2,-1, 1]),
                  'K': np.array([-1, 3, 0,-1,-3, 2, 1,-2, 0,-3,-3, 6,-2,-4,-1, 0,-1,-3,-2,-3]),
                  'M': np.array([-1,-2,-2,-4,-2, 0,-2,-3,-1, 2, 3,-2, 7, 0,-3,-2,-1,-1, 0, 1]),
                  'F': np.array([-3,-3,-4,-5,-2,-4,-3,-4,-1, 0, 1,-4, 0, 8,-4,-3,-2, 1, 4,-1]),
                  'P': np.array([-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,-1,-1,-4,-3,-3]),
                  'S': np.array([ 1,-1, 1, 0,-1, 0,-1, 0,-1,-3,-3, 0,-2,-3,-1, 5, 2,-4,-2,-2]),
                  'T': np.array([ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 2, 5,-3,-2, 0]),
                  'W': np.array([-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1, 1,-4,-4,-3,15, 2,-3]),
                  'Y': np.array([-2,-1,-2,-3,-3,-1,-2,-3, 2,-1,-1,-2, 0, 4,-3,-2,-2, 2, 8,-1]),
                  'V': np.array([ 0,-3,-3,-4,-1,-3,-3,-4,-4, 4, 1,-3, 1,-1,-3,-2, 0,-3,-1, 5])}

    elif name == 'blosum50':
        scheme = {'A': np.array([ 5,-2,-1,-2,-1,-1,-1, 0,-2,-1,-2,-1,-1,-3,-1, 1, 0,-3,-2, 0,-2,-1,-1,-5]),
                  'R': np.array([-2, 7,-1,-2,-4, 1, 0,-3, 0,-4,-3, 3,-2,-3,-3,-1,-1,-3,-1,-3,-1, 0,-1,-5]),
                  'N': np.array([-1,-1, 7, 2,-2, 0, 0, 0, 1,-3,-4, 0,-2,-4,-2, 1, 0,-4,-2,-3, 4, 0,-1,-5]),
                  'D': np.array([-2,-2, 2, 8,-4, 0, 2,-1,-1,-4,-4,-1,-4,-5,-1, 0,-1,-5,-3,-4, 5, 1,-1,-5]),
                  'C': np.array([-1,-4,-2,-4,13,-3,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-5,-3,-1,-3,-3,-2,-5]),
                  'Q': np.array([-1, 1, 0, 0,-3, 7, 2,-2, 1,-3,-2, 2, 0,-4,-1, 0,-1,-1,-1,-3, 0, 4,-1,-5]),
                  'E': np.array([-1, 0, 0, 2,-3, 2, 6,-3, 0,-4,-3, 1,-2,-3,-1,-1,-1,-3,-2,-3, 1, 5,-1,-5]),
                  'G': np.array([ 0,-3, 0,-1,-3,-2,-3, 8,-2,-4,-4,-2,-3,-4,-2, 0,-2,-3,-3,-4,-1,-2,-2,-5]),
                  'H': np.array([-2, 0, 1,-1,-3, 1, 0,-2,10,-4,-3, 0,-1,-1,-2,-1,-2,-3, 2,-4, 0, 0,-1,-5]),
                  'I': np.array([-1,-4,-3,-4,-2,-3,-4,-4,-4, 5, 2,-3, 2, 0,-3,-3,-1,-3,-1, 4,-4,-3,-1,-5]),
                  'L': np.array([-2,-3,-4,-4,-2,-2,-3,-4,-3, 2, 5,-3, 3, 1,-4,-3,-1,-2,-1, 1,-4,-3,-1,-5]),
                  'K': np.array([-1, 3, 0,-1,-3, 2, 1,-2, 0,-3,-3, 6,-2,-4,-1, 0,-1,-3,-2,-3, 0, 1,-1,-5]),
                  'M': np.array([-1,-2,-2,-4,-2, 0,-2,-3,-1, 2, 3,-2, 7, 0,-3,-2,-1,-1, 0, 1,-3,-1,-1,-5]),
                  'F': np.array([-3,-3,-4,-5,-2,-4,-3,-4,-1, 0, 1,-4, 0, 8,-4,-3,-2, 1, 4,-1,-4,-4,-2,-5]),
                  'P': np.array([-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,-1,-1,-4,-3,-3,-2,-1,-2,-5]),
                  'S': np.array([ 1,-1, 1, 0,-1, 0,-1, 0,-1,-3,-3, 0,-2,-3,-1, 5, 2,-4,-2,-2, 0, 0,-1,-5]),
                  'T': np.array([ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 2, 5,-3,-2, 0, 0,-1, 0,-5]),
                  'W': np.array([-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1, 1,-4,-4,-3,15, 2,-3,-5,-2,-3,-5]),
                  'Y': np.array([-2,-1,-2,-3,-3,-1,-2,-3, 2,-1,-1,-2, 0, 4,-3,-2,-2, 2, 8,-1,-3,-2,-1,-5]),
                  'V': np.array([ 0,-3,-3,-4,-1,-3,-3,-4,-4, 4, 1,-3, 1,-1,-3,-2, 0,-3,-1, 5,-4,-3,-1,-5]),
                  'B': np.array([-2,-1, 4, 5,-3, 0, 1,-1, 0,-4,-4, 0,-3,-4,-2, 0, 0,-5,-3,-4, 5, 2,-1,-5]),
                  'Z': np.array([-1, 0, 0, 1,-3, 4, 5,-2, 0,-3,-3, 1,-1,-4,-1, 0,-1,-2,-2,-3, 2, 5,-1,-5]),
                  'X': np.array([-1,-1,-1,-1,-2,-1,-1,-2,-1,-1,-1,-1,-1,-2,-2,-1, 0,-3,-1,-1,-1,-1,-1,-5]),
                  '*': np.array([-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5, 1])}

    elif name == 'one_hot':
        scheme = {'A': np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                  'R': np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                  'N': np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                  'D': np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                  'C': np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                  'Q': np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                  'E': np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                  'G': np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                  'H': np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]),
                  'I': np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]),
                  'L': np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]),
                  'K': np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]),
                  'M': np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
                  'F': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]),
                  'P': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]),
                  'S': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]),
                  'T': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
                  'W': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]),
                  'Y': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]),
                  'V': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]),
                  'X': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])}

    elif name == 'one_hot_20aa':
        scheme = {'A': np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                  'R': np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                  'N': np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                  'D': np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                  'C': np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                  'Q': np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                  'E': np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]),
                  'G': np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]),
                  'H': np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]),
                  'I': np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]),
                  'L': np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]),
                  'K': np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
                  'M': np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]),
                  'F': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]),
                  'P': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]),
                  'S': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
                  'T': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]),
                  'W': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]),
                  'Y': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]),
                  'V': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])}

    elif name == 'amino_to_idx':
        scheme = {'A': np.array([ 1]),
                  'R': np.array([ 2]),
                  'N': np.array([ 3]),
                  'D': np.array([ 4]),
                  'C': np.array([ 5]),
                  'Q': np.array([ 6]),
                  'E': np.array([ 7]),
                  'G': np.array([ 8]),
                  'H': np.array([ 9]),
                  'I': np.array([10]),
                  'L': np.array([11]),
                  'K': np.array([12]),
                  'M': np.array([13]),
                  'F': np.array([14]),
                  'P': np.array([15]),
                  'S': np.array([16]),
                  'T': np.array([17]),
                  'W': np.array([18]),
                  'Y': np.array([19]),
                  'V': np.array([20]),
                  'X': np.array([21])}

    elif name == 'phys_chem':
        scheme = {'A': np.array([   1,  -6.7, 0, 0,  0]),
                  'R': np.array([   4, -11.7, 0, 0,  0]),
                  'N': np.array([6.13,  51.5, 4, 0,  1]),
                  'D': np.array([4.77,  36.8, 2, 0,  1]),
                  'C': np.array([2.95,  20.1, 2, 2,  0]),
                  'Q': np.array([4.43, -14.4, 0, 0,  0]),
                  'E': np.array([2.78,  38.5, 1, 4, -1]),
                  'G': np.array([5.89, -15.5, 0, 0,  0]),
                  'H': np.array([2.43,  -8.4, 0, 0,  0]),
                  'I': np.array([2.72,   0.8, 0, 0,  0]),
                  'L': np.array([3.95,  17.2, 2, 2,  0]),
                  'K': np.array([ 1.6,  -2.5, 1, 2,  0]),
                  'M': np.array([3.78,  34.3, 1, 4, -1]),
                  'F': np.array([ 2.6,    -5, 1, 2,  0]),
                  'P': np.array([   0,  -4.2, 0, 0,  0]),
                  'S': np.array([8.08,  -7.9, 1, 0,  0]),
                  'T': np.array([4.66,  12.6, 1, 1,  0]),
                  'W': np.array([6.47,   2.9, 1, 1,  0]),
                  'Y': np.array([   4,   -13, 0, 0,  0]),
                  'V': np.array([   3, -10.9, 0, 0,  0])}

    elif name == 'blosum62':
        scheme = {'A': np.array([ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0, 0]),
                  'R': np.array([-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3,-1]),
                  'N': np.array([-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3,-1]),
                  'D': np.array([-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3,-1]),
                  'C': np.array([ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-2]),
                  'Q': np.array([-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2,-1]),
                  'E': np.array([-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2,-1]),
                  'G': np.array([ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3,-1]),
                  'H': np.array([-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3,-1]),
                  'I': np.array([-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3,-1]),
                  'L': np.array([-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,-1]),
                  'K': np.array([-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,-1]),
                  'M': np.array([-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1,-1]),
                  'F': np.array([-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1,-1]),
                  'P': np.array([-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2,-2]),
                  'S': np.array([ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2, 0]),
                  'T': np.array([ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0, 0]),
                  'W': np.array([-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3,-2]),
                  'Y': np.array([-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1,-1]),
                  'V': np.array([ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4,-1]),
                  'X': np.array([ 0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2, 0, 0,-2,-1,-1,-1])}

    elif name == 'blosum62_20aa':
        scheme = {'A': np.array([ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0]),
                  'R': np.array([-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3]),
                  'N': np.array([-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3]),
                  'D': np.array([-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3]),
                  'C': np.array([ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1]),
                  'Q': np.array([-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2]),
                  'E': np.array([-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2]),
                  'G': np.array([ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3]),
                  'H': np.array([-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3]),
                  'I': np.array([-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3]),
                  'L': np.array([-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1]),
                  'K': np.array([-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1]),
                  'M': np.array([-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1]),
                  'F': np.array([-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1]),
                  'P': np.array([-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2]),
                  'S': np.array([ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2]),
                  'T': np.array([ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0]),
                  'W': np.array([-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3]),
                  'Y': np.array([-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1]),
                  'V': np.array([ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4])}

    else:
        try:
            raise ValueError('Unknown encoding scheme')
        except ValueError as error:
            print(error)
            sys.exit(1)

    return scheme

def get_encoding_scheme_info(scheme):
    amino_acid_encoded = next(iter(scheme.values()))
    count_features = len(amino_acid_encoded)
    data_type = amino_acid_encoded.dtype

    return (count_features, data_type)

def encode_peptide(sequence,
                   encoding_scheme_name = 'blosum50_20aa'):
    sequence = sequence.upper()
    encoding_scheme = get_encoding_scheme(name = encoding_scheme_name)
    length_sequence = len(sequence)
    count_features, data_type = get_encoding_scheme_info(encoding_scheme)

    # Initialize encoded sequence array
    sequence_encoded = np.empty(shape = (length_sequence,
                                         count_features),
                                dtype = data_type)

    # Encode sequence
    for amino_acid_id in range(length_sequence):
        amino_acid = sequence[amino_acid_id]
        try:
            amino_acid_encoded = encoding_scheme[amino_acid]
        except KeyError:
            print('Unknown amino acid')
            sys.exit(1)

        sequence_encoded[amino_acid_id] = amino_acid_encoded

    return sequence_encoded


def make_peptide_encoding_dict(sequences,
                               encoding_scheme_name = 'blosum50_20aa'):
    peptide_encoding_dict = dict()
    for sequence in sequences:
        if sequence not in peptide_encoding_dict:
            peptide_encoding_dict[sequence] = encode_peptide(sequence,
                                                             encoding_scheme_name)

    return peptide_encoding_dict


def pad_sequences(sequence_array,
                  length_sequence_max = None,
                  padding_value = 0,
                  padding_side = 'right',
                  truncating_side = 'right'):

    length_sequence_array = len(sequence_array)

    # Get info from first sequence
    sequence = sequence_array[0]
    count_features = sequence.shape[1]
    data_type = sequence.dtype

    # Get max sequence length
    if length_sequence_max is None:
        length_sequence_max = max([sequence.shape[0] for sequence in sequence_array])

    # Initialize array of padded sequences
    sequence_padded_array = np.full(shape = (length_sequence_array,
                                             length_sequence_max,
                                             count_features),
                                    fill_value = padding_value,
                                    dtype = data_type)

    # Pad/truncate sequences
    for sequence_id in range(length_sequence_array):
        sequence = sequence_array[sequence_id]
        length_sequence = sequence.shape[0]

        # Padding
        if length_sequence <= length_sequence_max:
            if padding_side == 'right':
                sequence_padded_array[sequence_id,
                                      :length_sequence] = sequence
            elif padding_side == 'left':
                sequence_padded_array[sequence_id,
                                      length_sequence_max - length_sequence:] = sequence
            else:
                try:
                    raise ValueError("padding_side must be 'left' or 'right'")
                except ValueError as error:
                    print(error)
                    sys.exit(1)

        # Truncating
        else:
            if truncating_side == 'right':
                sequence_padded_array[sequence_id] = sequence[:length_sequence_max]
            elif truncating_side == 'left':
                sequence_padded_array[sequence_id] = sequence[length_sequence - length_sequence_max:]
            else:
                try:
                    raise ValueError("truncating_side must be 'left' or 'right'")
                except ValueError as error:
                    print(error)
                    sys.exit(1)

    return sequence_padded_array

def encode_unique_sequences(df,
                            encoding_scheme_name = 'blosum50_20aa',
                            sample_weight = False):
    unique_dict = dict()
    peptide_to_id_dict = dict()
    peptide_index = 0
    if sample_weight:
        peptide_count_array = list()

    for peptide in df['peptide']:
        if peptide not in peptide_to_id_dict:
            peptide_to_id_dict[peptide] = peptide_index
            peptide_index += 1
            if sample_weight:
                peptide_count_array.append(1)
        else:
            if sample_weight:
                peptide_count_array[peptide_to_id_dict[peptide]] += 1

    unique_dict['peptide_to_id_dict'] = peptide_to_id_dict
    if sample_weight:
        peptide_unique_count = len(peptide_to_id_dict)
        entries_count = df.shape[0]

        peptide_count_array = np.array(peptide_count_array)
        peptide_weight_array = np.log2(entries_count/(peptide_count_array))/np.log2(peptide_unique_count)
        peptide_weight_array = peptide_weight_array*(entries_count/np.sum(peptide_weight_array*peptide_count_array))
        unique_dict['peptide_weight_array'] = peptide_weight_array

    return unique_dict

def get_model_input(df,
                    df_peptides_unique,
                    peptides_unique_encoded,
                    use_embeddings = True,
                    encoding_scheme_name = None,
                    a1_max = None,
                    a2_max = None,
                    a3_max = None,
                    b1_max = None,
                    b2_max = None,
                    b3_max = None,
                    get_weights = False):
    """Prepares the embedding for the input features to the model"""
    # Initialize
    model_input = dict()

    if get_weights:
        model_input['weight'] = (df['peptide']
                                 .map(lambda x: (df_peptides_unique
                                                 .loc[x,
                                                      'weight']))
                                 .to_numpy())

    encoded_pep_ids = (df['peptide']
                       .map(lambda x: (df_peptides_unique
                                       .loc[x,
                                            'row_number']))
                       .to_list())

    model_input['peptide'] = peptides_unique_encoded[encoded_pep_ids]

    model_input['target'] = df['binder'].to_numpy()

    # Map negative TCR IDs to positive TCR IDs
    positive_binder_original_ids = df.query('binder == 1').index
    dict_original_ids_to_positive_tcr_array_ids = {positive_binder_original_ids[i]: i for i in range(len(positive_binder_original_ids))}
    list_original_positive_tcr_array_ids = [dict_original_ids_to_positive_tcr_array_ids[original_index] for original_index in df['original_index']]

    # Get embedded/encoded CDRs
    cdr_name_tuple = ('a1', 'a2', 'a3', 'b1', 'b2', 'b3')
    cdr_length_max_tuple = (a1_max,
                                 a2_max,
                                 a3_max,
                                 b1_max,
                                 b2_max,
                                 b3_max)
    if use_embeddings:
        # Load embedded positive CDRs
        file_embeddings = np.load(file = '../data/embedding.npz')

    for i in range(len(cdr_name_tuple)):
        cdr_name = cdr_name_tuple[i]
        cdr_length_max = cdr_length_max_tuple[i]
        if use_embeddings:
            # Retrive embedded positive CDRs
            model_input[cdr_name] = file_embeddings[cdr_name]
        else:
             # Encode positive CDRs
             model_input[cdr_name] = (df
                                      .query('binder == 1')[cdr_name.upper()]
                                      .map(lambda x: encode_peptide(sequence = x,
                                                                    encoding_scheme_name = encoding_scheme_name)))
             # Pad positive CDRs
             model_input[cdr_name] = pad_sequences(sequence_array = model_input[cdr_name].tolist(),
                                                   length_sequence_max = cdr_length_max,
                                                   padding_value = -5)/5

        # Get negative TCRs from the positive
        model_input[cdr_name] = model_input[cdr_name][list_original_positive_tcr_array_ids]

    return model_input

def adjust_batch_size(obs, batch_size, threshold = 0.5):
    if obs/batch_size < threshold:
        pass
    
    else:
        if (obs/batch_size % 1) >= threshold:
            pass
        else:
            while (obs/batch_size % 1) < threshold and (obs/batch_size % 1) != 0:
                batch_size += 1
    return batch_size
