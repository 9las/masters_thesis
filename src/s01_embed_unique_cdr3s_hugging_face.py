#!/usr/bin/env python
import numpy as np
import pandas as pd
import argparse
from transformers import AutoTokenizer, TFAutoModel, AutoModel
import datasets
import s99_project_functions
import os

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config")
args = parser.parse_args()
config_filename_cdr3 = args.config

# Load config
config_cdr3 = s99_project_functions.load_config(config_filename_cdr3)
config_main = s99_project_functions.load_config('s97_main_config.yaml')

# Set parameters from config
embedder_name_cdr3 = config_cdr3['default']['embedder_name_cdr3']
embedder_index_cdr3 = config_cdr3['default']['embedder_index_cdr3']
embedder_batch_size_cdr3 = config_cdr3['default']['embedder_batch_size_cdr3']
embedder_backend_cdr3 = config_cdr3['default']['embedder_backend_cdr3']
data_filename = config_main['default']['data_filename']

# Read data
features = datasets.Features({'A3' : datasets.Value('string'),
                              'B3' : datasets.Value('string'),
                              'binder': datasets.Value('bool'),
                              'original_index': datasets.Value('int32')})

data = datasets.load_dataset(path = 'csv',
                             data_files = os.path.join('../data/raw',
                                                       data_filename),
                             features = features)

# Define functions
def get_cdr3_length(example):
    return_dict = dict()
    return_dict['a3_length'] = len(example['A3'])
    return_dict['b3_length'] = len(example['B3'])

    return return_dict

def split_amino_acids(example):
    example['A3'] = ' '.join(example['A3'])
    example['B3'] = ' '.join(example['B3'])

    return example

def add_embeddings(batch):
    embeddings_dict = dict()

    cdr3_name_tuple = ('A3',
                       'B3')

    cdr3_encoded_name_tuple = ('a3_encoded',
                               'b3_encoded')

    for i in range(len(cdr3_name_tuple)):
        cdr3_name = cdr3_name_tuple[i]
        cdr3_encoded_name = cdr3_encoded_name_tuple[i]

        embeddings_dict[cdr3_encoded_name] = model(**tokenizer(text = batch[cdr3_name],
                                                               return_tensors = embedder_backend_cdr3,
                                                               padding = 'longest'))

        embeddings_dict[cdr3_encoded_name] = embeddings_dict[cdr3_encoded_name]['last_hidden_state']

    return embeddings_dict

# Get the embedder
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = embedder_name_cdr3)

if embedder_backend_cdr3  == 'tf':
    model = TFAutoModel.from_pretrained(pretrained_model_name_or_path = embedder_name_cdr3)
elif embedder_backend_cdr3 == 'pt':
    model = AutoModel.from_pretrained(pretrained_model_name_or_path = embedder_name_cdr3)

# Do the embedding for positive binders - all unique TCRs
data = (data
        .filter(lambda x: x['binder'] is True)
        .map(function = get_cdr3_length)
        .map(function = split_amino_acids)
        .map(add_embeddings,
             batched = True,
             batch_size = embedder_batch_size_cdr3))

# Make dataset into a pandas dataframe
data.set_format(type = 'pandas',
                columns = ['original_index',
                           'a3_encoded',
                           'b3_encoded',
                           'a3_length',
                           'b3_length'])

data_df = data['train'][:]

data_df = (data_df
           .set_index(keys = 'original_index'))

# Make embeddings into ND arrays
data_df[['a3_encoded',
         'b3_encoded']] = (data_df[['a3_encoded',
                                    'b3_encoded']]
                           .applymap(func = lambda x: np.vstack(x)))

# Remove padding
data_df['a3_encoded'] = (data_df
                         .apply(func = lambda x: x['a3_encoded'][1:x['a3_length'] + 1], # Add one due to <CLS> token
                         axis = 1))
data_df['b3_encoded'] = (data_df
                         .apply(func = lambda x: x['b3_encoded'][1:x['b3_length'] + 1],
                         axis = 1))

# Save embeddings
data_df.to_pickle(path = '../data/s01_e3c{}_embedding.pkl'.format(embedder_index_cdr3))
