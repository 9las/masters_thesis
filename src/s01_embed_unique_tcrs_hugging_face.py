#!/usr/bin/env python
import numpy as np
import pandas as pd
import argparse
from transformers import AutoTokenizer, TFAutoModel
import datasets
import s99_project_functions
import os

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config")
args = parser.parse_args()
config_filename_tcr = args.config

# Load config
config_tcr = s99_project_functions.load_config(config_filename_tcr)
config_main = s99_project_functions.load_config('s97_main_config.yaml')

# Set parameters from config
embedder_name_tcr = config_tcr['default']['embedder_name_tcr']
embedder_index_tcr = config_tcr['default']['embedder_index_tcr']
embedder_batch_size_tcr = config_tcr['default']['embedder_batch_size_tcr']
data_filename = config_main['default']['data_filename']

# Read data
features = datasets.Features({'TRA_aa' : datasets.Value('string'),
                              'TRB_aa' : datasets.Value('string'),
                              'A1_start': datasets.Value('int16'),
                              'A1_end': datasets.Value('int16'),
                              'A2_start': datasets.Value('int16'),
                              'A2_end': datasets.Value('int16'),
                              'A3_start': datasets.Value('int16'),
                              'A3_end': datasets.Value('int16'),
                              'B1_start': datasets.Value('int16'),
                              'B1_end': datasets.Value('int16'),
                              'B2_start': datasets.Value('int16'),
                              'B2_end': datasets.Value('int16'),
                              'B3_start': datasets.Value('int16'),
                              'B3_end': datasets.Value('int16'),
                              'binder': datasets.Value('bool'),
                              'original_index': datasets.Value('int32')})

data = datasets.load_dataset(path = 'csv',
                             data_files = os.path.join('../data/raw',
                                                       data_filename),
                             features = features)

# Define functions
def add_embeddings(batch):
    embeddings_dict = dict()

    tcr_name_tuple = ('TRA_aa',
                      'TRB_aa')

    tcr_encoded_name_tuple = ('tra_aa_encoded',
                              'trb_aa_encoded')

    for i in range(len(tcr_name_tuple)):
        tcr_name = tcr_name_tuple[i]
        tcr_encoded_name = tcr_encoded_name_tuple[i]

        embeddings_dict[tcr_encoded_name] = model(**tokenizer(text = batch[tcr_name],
                                                              return_tensors = 'tf',
                                                              padding = 'longest'))

        embeddings_dict[tcr_encoded_name] = embeddings_dict[tcr_encoded_name]['last_hidden_state']

    return embeddings_dict

# Get the embedder
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = embedder_name_tcr)
model = TFAutoModel.from_pretrained(pretrained_model_name_or_path = embedder_name_tcr)

# Do the embedding for positive binders - all unique TCRs
data = (data
        .filter(lambda x: x['binder'] is True)
        .map(add_embeddings,
                batched = True,
                batch_size = embedder_batch_size_tcr))

# Make dataset into a pandas dataframe
data.set_format(type = 'pandas',
                columns = ['original_index',
                           'tra_aa_encoded',
                           'trb_aa_encoded',
                           'A1_start',
                           'A1_end',
                           'A2_start',
                           'A2_end',
                           'A3_start',
                           'A3_end',
                           'B1_start',
                           'B1_end',
                           'B2_start',
                           'B2_end',
                           'B3_start',
                           'B3_end'])

data_df = data['train'][:]

data_df = (data_df
           .set_index(keys = 'original_index'))

data_df[['tra_aa_encoded',
         'trb_aa_encoded']] = (data_df[['tra_aa_encoded',
                                        'trb_aa_encoded']]
                                        .applymap(func = lambda x: np.vstack(x)))

# Extract the CDRs
cdr_name_tuple = ('a1_encoded',
                  'a2_encoded',
                  'a3_encoded',
                  'b1_encoded',
                  'b2_encoded',
                  'b3_encoded')

tcr_name_tuple = ('tra_aa_encoded',
                  'tra_aa_encoded',
                  'tra_aa_encoded',
                  'trb_aa_encoded',
                  'trb_aa_encoded',
                  'trb_aa_encoded')

cdr_start_name_tuple = ('A1_start',
                        'A2_start',
                        'A3_start',
                        'B1_start',
                        'B2_start',
                        'B3_start')

cdr_end_name_tuple = ('A1_end',
                      'A2_end',
                      'A3_end',
                      'B1_end',
                      'B2_end',
                      'B3_end')

for i in range(len(cdr_name_tuple)):
    cdr_name = cdr_name_tuple[i]
    tcr_name = tcr_name_tuple[i]
    cdr_start_name = cdr_start_name_tuple[i]
    cdr_end_name = cdr_end_name_tuple[i]

    data_df[cdr_name] = data_df.apply(func = lambda x: x[tcr_name][x[cdr_start_name] + 1: x[cdr_end_name] + 1], # Add one due to <CLS> token
                                      axis = 1)

# Make embeddings into ND arrays
data_df = (data_df
           .filter(items = cdr_name_tuple)
           .applymap(func = lambda x: np.vstack(x)))


# Save embeddings
data_df.to_pickle(path = '../data/s01_et{}_embedding.pkl'.format(embedder_index_tcr))
