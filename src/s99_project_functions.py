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
import tensorflow as tf
from sklearn.metrics import roc_auc_score

def mkdir(outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

# Function to load yaml configuration file
def load_config(config_name):
    with open(config_name) as file:
        config = yaml.safe_load(file)

    return config

#Custom metric for AUC 0.1
def auc01_numpy(y_true, y_pred):    
    try:
        auc01 = roc_auc_score(y_true, y_pred, max_fpr = 0.1)
    except ValueError:
        #Exception for when a positive observation is not present in a batch
        auc01 = np.array([float(0)])
    return auc01

def auc01(y_true, y_pred):
    "Allows Tensorflow to use the function during training"
    auc01 = tf.numpy_function(auc01_numpy, [y_true, y_pred], tf.float64)
    return auc01

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
        scheme = {'A': np.array(object = [ 5,-2,-1,-2,-1,-1,-1, 0,-2,-1,-2,-1,-1,-3,-1, 1, 0,-3,-2, 0],
                                dtype = np.byte),
                  'R': np.array(object = [-2, 7,-1,-2,-4, 1, 0,-3, 0,-4,-3, 3,-2,-3,-3,-1,-1,-3,-1,-3],
                                dtype = np.byte),
                  'N': np.array(object = [-1,-1, 7, 2,-2, 0, 0, 0, 1,-3,-4, 0,-2,-4,-2, 1, 0,-4,-2,-3],
                                dtype = np.byte),
                  'D': np.array(object = [-2,-2, 2, 8,-4, 0, 2,-1,-1,-4,-4,-1,-4,-5,-1, 0,-1,-5,-3,-4],
                                dtype = np.byte),
                  'C': np.array(object = [-1,-4,-2,-4,13,-3,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-5,-3,-1],
                                dtype = np.byte),
                  'Q': np.array(object = [-1, 1, 0, 0,-3, 7, 2,-2, 1,-3,-2, 2, 0,-4,-1, 0,-1,-1,-1,-3],
                                dtype = np.byte),
                  'E': np.array(object = [-1, 0, 0, 2,-3, 2, 6,-3, 0,-4,-3, 1,-2,-3,-1,-1,-1,-3,-2,-3],
                                dtype = np.byte),
                  'G': np.array(object = [ 0,-3, 0,-1,-3,-2,-3, 8,-2,-4,-4,-2,-3,-4,-2, 0,-2,-3,-3,-4],
                                dtype = np.byte),
                  'H': np.array(object = [-2, 0, 1,-1,-3, 1, 0,-2,10,-4,-3, 0,-1,-1,-2,-1,-2,-3, 2,-4],
                                dtype = np.byte),
                  'I': np.array(object = [-1,-4,-3,-4,-2,-3,-4,-4,-4, 5, 2,-3, 2, 0,-3,-3,-1,-3,-1, 4],
                                dtype = np.byte),
                  'L': np.array(object = [-2,-3,-4,-4,-2,-2,-3,-4,-3, 2, 5,-3, 3, 1,-4,-3,-1,-2,-1, 1],
                                dtype = np.byte),
                  'K': np.array(object = [-1, 3, 0,-1,-3, 2, 1,-2, 0,-3,-3, 6,-2,-4,-1, 0,-1,-3,-2,-3],
                                dtype = np.byte),
                  'M': np.array(object = [-1,-2,-2,-4,-2, 0,-2,-3,-1, 2, 3,-2, 7, 0,-3,-2,-1,-1, 0, 1],
                                dtype = np.byte),
                  'F': np.array(object = [-3,-3,-4,-5,-2,-4,-3,-4,-1, 0, 1,-4, 0, 8,-4,-3,-2, 1, 4,-1],
                                dtype = np.byte),
                  'P': np.array(object = [-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,-1,-1,-4,-3,-3],
                                dtype = np.byte),
                  'S': np.array(object = [ 1,-1, 1, 0,-1, 0,-1, 0,-1,-3,-3, 0,-2,-3,-1, 5, 2,-4,-2,-2],
                                dtype = np.byte),
                  'T': np.array(object = [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 2, 5,-3,-2, 0],
                                dtype = np.byte),
                  'W': np.array(object = [-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1, 1,-4,-4,-3,15, 2,-3],
                                dtype = np.byte),
                  'Y': np.array(object = [-2,-1,-2,-3,-3,-1,-2,-3, 2,-1,-1,-2, 0, 4,-3,-2,-2, 2, 8,-1],
                                dtype = np.byte),
                  'V': np.array(object = [ 0,-3,-3,-4,-1,-3,-3,-4,-4, 4, 1,-3, 1,-1,-3,-2, 0,-3,-1, 5],
                                dtype = np.byte)}

    elif name == 'blosum50':
        scheme = {'A': np.array(object = [ 5,-2,-1,-2,-1,-1,-1, 0,-2,-1,-2,-1,-1,-3,-1, 1, 0,-3,-2, 0,-2,-1,-1,-5],
                                dtype = np.byte),
                  'R': np.array(object = [-2, 7,-1,-2,-4, 1, 0,-3, 0,-4,-3, 3,-2,-3,-3,-1,-1,-3,-1,-3,-1, 0,-1,-5],
                                dtype = np.byte),
                  'N': np.array(object = [-1,-1, 7, 2,-2, 0, 0, 0, 1,-3,-4, 0,-2,-4,-2, 1, 0,-4,-2,-3, 4, 0,-1,-5],
                                dtype = np.byte),
                  'D': np.array(object = [-2,-2, 2, 8,-4, 0, 2,-1,-1,-4,-4,-1,-4,-5,-1, 0,-1,-5,-3,-4, 5, 1,-1,-5],
                                dtype = np.byte),
                  'C': np.array(object = [-1,-4,-2,-4,13,-3,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-5,-3,-1,-3,-3,-2,-5],
                                dtype = np.byte),
                  'Q': np.array(object = [-1, 1, 0, 0,-3, 7, 2,-2, 1,-3,-2, 2, 0,-4,-1, 0,-1,-1,-1,-3, 0, 4,-1,-5],
                                dtype = np.byte),
                  'E': np.array(object = [-1, 0, 0, 2,-3, 2, 6,-3, 0,-4,-3, 1,-2,-3,-1,-1,-1,-3,-2,-3, 1, 5,-1,-5],
                                dtype = np.byte),
                  'G': np.array(object = [ 0,-3, 0,-1,-3,-2,-3, 8,-2,-4,-4,-2,-3,-4,-2, 0,-2,-3,-3,-4,-1,-2,-2,-5],
                                dtype = np.byte),
                  'H': np.array(object = [-2, 0, 1,-1,-3, 1, 0,-2,10,-4,-3, 0,-1,-1,-2,-1,-2,-3, 2,-4, 0, 0,-1,-5],
                                dtype = np.byte),
                  'I': np.array(object = [-1,-4,-3,-4,-2,-3,-4,-4,-4, 5, 2,-3, 2, 0,-3,-3,-1,-3,-1, 4,-4,-3,-1,-5],
                                dtype = np.byte),
                  'L': np.array(object = [-2,-3,-4,-4,-2,-2,-3,-4,-3, 2, 5,-3, 3, 1,-4,-3,-1,-2,-1, 1,-4,-3,-1,-5],
                                dtype = np.byte),
                  'K': np.array(object = [-1, 3, 0,-1,-3, 2, 1,-2, 0,-3,-3, 6,-2,-4,-1, 0,-1,-3,-2,-3, 0, 1,-1,-5],
                                dtype = np.byte),
                  'M': np.array(object = [-1,-2,-2,-4,-2, 0,-2,-3,-1, 2, 3,-2, 7, 0,-3,-2,-1,-1, 0, 1,-3,-1,-1,-5],
                                dtype = np.byte),
                  'F': np.array(object = [-3,-3,-4,-5,-2,-4,-3,-4,-1, 0, 1,-4, 0, 8,-4,-3,-2, 1, 4,-1,-4,-4,-2,-5],
                                dtype = np.byte),
                  'P': np.array(object = [-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,-1,-1,-4,-3,-3,-2,-1,-2,-5],
                                dtype = np.byte),
                  'S': np.array(object = [ 1,-1, 1, 0,-1, 0,-1, 0,-1,-3,-3, 0,-2,-3,-1, 5, 2,-4,-2,-2, 0, 0,-1,-5],
                                dtype = np.byte),
                  'T': np.array(object = [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 2, 5,-3,-2, 0, 0,-1, 0,-5],
                                dtype = np.byte),
                  'W': np.array(object = [-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1, 1,-4,-4,-3,15, 2,-3,-5,-2,-3,-5],
                                dtype = np.byte),
                  'Y': np.array(object = [-2,-1,-2,-3,-3,-1,-2,-3, 2,-1,-1,-2, 0, 4,-3,-2,-2, 2, 8,-1,-3,-2,-1,-5],
                                dtype = np.byte),
                  'V': np.array(object = [ 0,-3,-3,-4,-1,-3,-3,-4,-4, 4, 1,-3, 1,-1,-3,-2, 0,-3,-1, 5,-4,-3,-1,-5],
                                dtype = np.byte),
                  'B': np.array(object = [-2,-1, 4, 5,-3, 0, 1,-1, 0,-4,-4, 0,-3,-4,-2, 0, 0,-5,-3,-4, 5, 2,-1,-5],
                                dtype = np.byte),
                  'Z': np.array(object = [-1, 0, 0, 1,-3, 4, 5,-2, 0,-3,-3, 1,-1,-4,-1, 0,-1,-2,-2,-3, 2, 5,-1,-5],
                                dtype = np.byte),
                  'X': np.array(object = [-1,-1,-1,-1,-2,-1,-1,-2,-1,-1,-1,-1,-1,-2,-2,-1, 0,-3,-1,-1,-1,-1,-1,-5],
                                dtype = np.byte),
                  '*': np.array(object = [-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5, 1],
                                dtype = np.byte)}

    elif name == 'one_hot':
        scheme = {'A': np.array(object = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'R': np.array(object = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'N': np.array(object = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'D': np.array(object = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'C': np.array(object = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'Q': np.array(object = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'E': np.array(object = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'G': np.array(object = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'H': np.array(object = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'I': np.array(object = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'L': np.array(object = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'K': np.array(object = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'M': np.array(object = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'F': np.array(object = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'P': np.array(object = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'S': np.array(object = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                                dtype = np.bool_),
                  'T': np.array(object = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                                dtype = np.bool_),
                  'W': np.array(object = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                                dtype = np.bool_),
                  'Y': np.array(object = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                                dtype = np.bool_),
                  'V': np.array(object = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                                dtype = np.bool_),
                  'X': np.array(object = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                                dtype = np.bool_)}

    elif name == 'one_hot_20aa':
        scheme = {'A': np.array(object = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'R': np.array(object = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'N': np.array(object = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'D': np.array(object = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'C': np.array(object = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'Q': np.array(object = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'E': np.array(object = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'G': np.array(object = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'H': np.array(object = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'I': np.array(object = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'L': np.array(object = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'K': np.array(object = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'M': np.array(object = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'F': np.array(object = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                                dtype = np.bool_),
                  'P': np.array(object = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                                dtype = np.bool_),
                  'S': np.array(object = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                                dtype = np.bool_),
                  'T': np.array(object = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                                dtype = np.bool_),
                  'W': np.array(object = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                                dtype = np.bool_),
                  'Y': np.array(object = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                                dtype = np.bool_),
                  'V': np.array(object = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                                dtype = np.bool_)}

    elif name == 'amino_to_idx':
        scheme = {'A': np.array(object = [ 1],
                                dtype = np.ubyte),
                  'R': np.array(object = [ 2],
                                dtype = np.ubyte),
                  'N': np.array(object = [ 3],
                                dtype = np.ubyte),
                  'D': np.array(object = [ 4],
                                dtype = np.ubyte),
                  'C': np.array(object = [ 5],
                                dtype = np.ubyte),
                  'Q': np.array(object = [ 6],
                                dtype = np.ubyte),
                  'E': np.array(object = [ 7],
                                dtype = np.ubyte),
                  'G': np.array(object = [ 8],
                                dtype = np.ubyte),
                  'H': np.array(object = [ 9],
                                dtype = np.ubyte),
                  'I': np.array(object = [10],
                                dtype = np.ubyte),
                  'L': np.array(object = [11],
                                dtype = np.ubyte),
                  'K': np.array(object = [12],
                                dtype = np.ubyte),
                  'M': np.array(object = [13],
                                dtype = np.ubyte),
                  'F': np.array(object = [14],
                                dtype = np.ubyte),
                  'P': np.array(object = [15],
                                dtype = np.ubyte),
                  'S': np.array(object = [16],
                                dtype = np.ubyte),
                  'T': np.array(object = [17],
                                dtype = np.ubyte),
                  'W': np.array(object = [18],
                                dtype = np.ubyte),
                  'Y': np.array(object = [19],
                                dtype = np.ubyte),
                  'V': np.array(object = [20],
                                dtype = np.ubyte),
                  'X': np.array(object = [21],
                                dtype = np.ubyte)}

    elif name == 'phys_chem':
        scheme = {'A': np.array(object = [   1,  -6.7, 0, 0,  0],
                                dtype = np.half),
                  'R': np.array(object = [   4, -11.7, 0, 0,  0],
                                dtype = np.half),
                  'N': np.array(object = [6.13,  51.5, 4, 0,  1],
                                dtype = np.half),
                  'D': np.array(object = [4.77,  36.8, 2, 0,  1],
                                dtype = np.half),
                  'C': np.array(object = [2.95,  20.1, 2, 2,  0],
                                dtype = np.half),
                  'Q': np.array(object = [4.43, -14.4, 0, 0,  0],
                                dtype = np.half),
                  'E': np.array(object = [2.78,  38.5, 1, 4, -1],
                                dtype = np.half),
                  'G': np.array(object = [5.89, -15.5, 0, 0,  0],
                                dtype = np.half),
                  'H': np.array(object = [2.43,  -8.4, 0, 0,  0],
                                dtype = np.half),
                  'I': np.array(object = [2.72,   0.8, 0, 0,  0],
                                dtype = np.half),
                  'L': np.array(object = [3.95,  17.2, 2, 2,  0],
                                dtype = np.half),
                  'K': np.array(object = [ 1.6,  -2.5, 1, 2,  0],
                                dtype = np.half),
                  'M': np.array(object = [3.78,  34.3, 1, 4, -1],
                                dtype = np.half),
                  'F': np.array(object = [ 2.6,    -5, 1, 2,  0],
                                dtype = np.half),
                  'P': np.array(object = [   0,  -4.2, 0, 0,  0],
                                dtype = np.half),
                  'S': np.array(object = [8.08,  -7.9, 1, 0,  0],
                                dtype = np.half),
                  'T': np.array(object = [4.66,  12.6, 1, 1,  0],
                                dtype = np.half),
                  'W': np.array(object = [6.47,   2.9, 1, 1,  0],
                                dtype = np.half),
                  'Y': np.array(object = [   4,   -13, 0, 0,  0],
                                dtype = np.half),
                  'V': np.array(object = [   3, -10.9, 0, 0,  0],
                                dtype = np.half)}

    elif name == 'blosum62':
        scheme = {'A': np.array(object = [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0, 0],
                                dtype = np.byte),
                  'R': np.array(object = [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3,-1],
                                dtype = np.byte),
                  'N': np.array(object = [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3,-1],
                                dtype = np.byte),
                  'D': np.array(object = [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3,-1],
                                dtype = np.byte),
                  'C': np.array(object = [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-2],
                                dtype = np.byte),
                  'Q': np.array(object = [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2,-1],
                                dtype = np.byte),
                  'E': np.array(object = [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2,-1],
                                dtype = np.byte),
                  'G': np.array(object = [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3,-1],
                                dtype = np.byte),
                  'H': np.array(object = [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3,-1],
                                dtype = np.byte),
                  'I': np.array(object = [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3,-1],
                                dtype = np.byte),
                  'L': np.array(object = [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,-1],
                                dtype = np.byte),
                  'K': np.array(object = [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,-1],
                                dtype = np.byte),
                  'M': np.array(object = [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1,-1],
                                dtype = np.byte),
                  'F': np.array(object = [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1,-1],
                                dtype = np.byte),
                  'P': np.array(object = [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2,-2],
                                dtype = np.byte),
                  'S': np.array(object = [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2, 0],
                                dtype = np.byte),
                  'T': np.array(object = [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0, 0],
                                dtype = np.byte),
                  'W': np.array(object = [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3,-2],
                                dtype = np.byte),
                  'Y': np.array(object = [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1,-1],
                                dtype = np.byte),
                  'V': np.array(object = [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4,-1],
                                dtype = np.byte),
                  'X': np.array(object = [ 0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2, 0, 0,-2,-1,-1,-1],
                                dtype = np.byte)}

    elif name == 'blosum62_20aa':
        scheme = {'A': np.array(object = [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
                                dtype = np.byte),
                  'R': np.array(object = [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
                                dtype = np.byte),
                  'N': np.array(object = [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
                                dtype = np.byte),
                  'D': np.array(object = [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
                                dtype = np.byte),
                  'C': np.array(object = [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
                                dtype = np.byte),
                  'Q': np.array(object = [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
                                dtype = np.byte),
                  'E': np.array(object = [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
                                dtype = np.byte),
                  'G': np.array(object = [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
                                dtype = np.byte),
                  'H': np.array(object = [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
                                dtype = np.byte),
                  'I': np.array(object = [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
                                dtype = np.byte),
                  'L': np.array(object = [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
                                dtype = np.byte),
                  'K': np.array(object = [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
                                dtype = np.byte),
                  'M': np.array(object = [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
                                dtype = np.byte),
                  'F': np.array(object = [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
                                dtype = np.byte),
                  'P': np.array(object = [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
                                dtype = np.byte),
                  'S': np.array(object = [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
                                dtype = np.byte),
                  'T': np.array(object = [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],
                                dtype = np.byte),
                  'W': np.array(object = [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],
                                dtype = np.byte),
                  'Y': np.array(object = [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
                                dtype = np.byte),
                  'V': np.array(object = [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],
                                dtype = np.byte)}

    else:
        try:
            raise ValueError('Unknown encoding scheme')
        except ValueError as error:
            print(error)
            sys.exit(1)

    return scheme

def get_encoding_scheme_info(scheme):
    amino_acid_encoded = next(iter(scheme.values()))
    embedding_size = len(amino_acid_encoded)
    data_type = amino_acid_encoded.dtype

    return (embedding_size, data_type)

def encode_sequence(sequence,
                   embedding_name = 'blosum50_20aa'):
    sequence = sequence.upper()
    encoding_scheme = get_encoding_scheme(name = embedding_name)
    length_sequence = len(sequence)
    embedding_size, data_type = get_encoding_scheme_info(encoding_scheme)

    # Initialize encoded sequence array
    sequence_encoded = np.empty(shape = (length_sequence,
                                         embedding_size),
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

def pad_sequence(sequence,
                 sequence_length_max,
                 padding_value = 0,
                 padding_side = 'right',
                 truncating_side = 'right'):

    sequence_length = sequence.shape[0]
    embedding_size = sequence.shape[1]
    data_type = sequence.dtype

    # Padding
    if sequence_length <= sequence_length_max:

        # Initialize padded sequence
        sequence_padded = np.full(shape = (sequence_length_max,
                                           embedding_size),
                                  fill_value = padding_value,
                                  dtype = data_type)

        if padding_side == 'right':
            sequence_padded[:sequence_length] = sequence
        elif padding_side == 'left':
            sequence_padded[sequence_length_max - sequence_length:] = sequence
        else:
            try:
                raise ValueError("padding_side must be 'left' or 'right'")
            except ValueError as error:
                print(error)
                sys.exit(1)

    # Truncating
    else:
        if truncating_side == 'right':
            sequence_padded = sequence[:sequence_length_max]
        elif truncating_side == 'left':
            sequence_padded = sequence[sequence_length - sequence_length_max:]
        else:
            try:
                raise ValueError("truncating_side must be 'left' or 'right'")
            except ValueError as error:
                print(error)
                sys.exit(1)

    return sequence_padded

def encode_unique_peptides(df,
                           encoding_name = 'blosum50_20aa'):
    df = (df
          .groupby('peptide')
          .agg(count = ('peptide',
                        'count'))
          .assign(peptide_encoded = lambda x: (x.index
                                               .map(lambda y: encode_sequence(sequence = y,
                                                                              embedding_name = encoding_name)))))

    return df

def encode_unique_tcrs(df,
                       encoding_name = 'blosum50_20aa'):
    # Get postive binders - all unique TCRs
    df = (df
          .query('binder == 1')
          .set_index('original_index'))

    # Do the encoding
    cdr_encoded_name_tuple = ('a1_encoded',
                              'a2_encoded',
                              'a3_encoded',
                              'b1_encoded',
                              'b2_encoded',
                              'b3_encoded')

    cdr_name_tuple = ('A1', 'A2', 'A3', 'B1', 'B2', 'B3')

    for i in range(len(cdr_name_tuple)):
        cdr_encoded_name = cdr_encoded_name_tuple[i]
        cdr_name = cdr_name_tuple[i]

        df[cdr_encoded_name] = (df[cdr_name]
                                .map(lambda x: encode_sequence(sequence = x,
                                                               embedding_name = encoding_name)))

    df = (df
          .filter(items = cdr_encoded_name_tuple))

    return df

def pad_unique_peptides(df,
                        padding_value = 0,
                        padding_side = 'right',
                        truncating_side = 'right'):

    peptide_length_max = df.index.map(len).max()

    df = (df
          .assign(peptide_encoded = lambda x: (x['peptide_encoded']
                                               .map(lambda y: pad_sequence(sequence = y,
                                                                           sequence_length_max = peptide_length_max,
                                                                           padding_value = padding_value,
                                                                           padding_side = padding_side,
                                                                           truncating_side = truncating_side)))))

    return df

def pad_unique_tcrs(df,
                    padding_value = 0,
                    padding_side = 'right',
                    truncating_side = 'right'):

    cdr_name_tuple = ('a1_encoded',
                      'a2_encoded',
                      'a3_encoded',
                      'b1_encoded',
                      'b2_encoded',
                      'b3_encoded')

    df_padded = df.copy()

    for cdr_name in cdr_name_tuple:
        cdr_length_max = df_padded[cdr_name].map(lambda x: x.shape[0]).max()

        df_padded[cdr_name] = (df_padded[cdr_name]
                               .map(lambda x: pad_sequence(sequence = x,
                                                           sequence_length_max = cdr_length_max,
                                                           padding_value = padding_value,
                                                           padding_side = padding_side,
                                                           truncating_side = truncating_side)))

    return df_padded

def get_peptide_map(df,
                    embedding_name = 'blosum50_20aa',
                    sample_weight = False,
                    padding_value = 0,
                    padding_side = 'right',
                    truncating_side = 'right'):

    # Calculate sample weights
    if sample_weight:
        peptide_map = (df
                       .groupby('peptide')
                       .agg(count = ('peptide',
                                     'count')))

        peptide_unique_count = peptide_map.shape[0]
        peptide_count = df.shape[0]

        peptide_map = (peptide_map
                       .assign(weight = lambda x: (np.log2(peptide_count
                                                           / x['count'])
                                                   / np.log2(peptide_unique_count)))
                       .assign(weight = lambda x: (x['weight']
                                                   * (peptide_count
                                                      / np.sum(x['weight']
                                                               * x['count']))))
                       .drop(labels = 'count',
                             axis = 1))
    else:
        peptide_map = (df
                       .groupby('peptide')
                       .agg(weight = ('peptide',
                                      lambda x: 1)))
    # Do the encoding
    if embedding_name in {'blosum50_20aa',
                          'blosum50',
                          'one_hot',
                          'one_hot_20aa',
                          'amino_to_idx',
                          'phys_chem',
                          'blosum62',
                          'blosum62_20aa'}:

        peptide_map = (peptide_map
                       .assign(peptide_encoded = lambda x: (x.index
                                                            .map(lambda y: encode_sequence(sequence = y,
                                                                                          embedding_name = embedding_name)))))
    else:
        print('hej')
    # Do the padding
    sequence_length_max = peptide_map.index.map(len).max()
    peptide_map = (peptide_map
                   .assign(peptide_encoded = lambda x: (x['peptide_encoded']
                                                        .map(lambda y: pad_sequence(sequence = y,
                                                                                    sequence_length_max = sequence_length_max,
                                                                                    padding_value = padding_value,
                                                                                    padding_side = padding_side,
                                                                                    truncating_side = truncating_side)))))

    return peptide_map

def encode_unique_peptides_(df,
                           embedding_name = 'blosum50_20aa',
                           sample_weight = False):
    out_dict = dict()
    out_dict['peptide_to_id_dict'] = dict()
    peptide_index = 0
    out_dict['peptide_encoded_array'] = list()
    if sample_weight:
        peptide_count_array = list()

    for peptide in df['peptide']:
        if peptide not in out_dict['peptide_to_id_dict']:
            out_dict['peptide_to_id_dict'][peptide] = peptide_index
            peptide_index += 1
            peptide_encoded = encode_sequence(sequence = peptide,
                                             embedding_name = embedding_name )
            out_dict['peptide_encoded_array'].append(peptide_encoded)
            if sample_weight:
                peptide_count_array.append(1)
        else:
            if sample_weight:
                peptide_id = out_dict['peptide_to_id_dict'][peptide]
                peptide_count_array[peptide_id] += 1

    out_dict['peptide_encoded_array'] = pad_sequences(sequence_array = out_dict['peptide_encoded_array'],
                                                      padding_value = -5)/5

    peptide_unique_count = len(out_dict['peptide_to_id_dict']) 

    if sample_weight:
        entries_count = df.shape[0]

        peptide_count_array = np.array(peptide_count_array)
        out_dict['peptide_weight_array'] = (np.log2(entries_count
                                                    / peptide_count_array)
                                            / np.log2(peptide_unique_count))
        out_dict['peptide_weight_array'] = (out_dict['peptide_weight_array']
                                            * (entries_count
                                               / np.sum(out_dict['peptide_weight_array']
                                                        * peptide_count_array)))
    else:
        out_dict['peptide_weight_array'] = np.ones(shape = peptide_unique_count,
                                                   dtype = np.ubyte)

    return out_dict

def get_model_input(df,
                    df_peptides_unique,
                    peptides_unique_encoded,
                    use_embeddings = True,
                    embedding_name = None,
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
                                      .map(lambda x: encode_sequence(sequence = x,
                                                                    embedding_name = embedding_name)))
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
