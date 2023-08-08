#!/usr/bin/env python
import numpy as np
import pandas as pd
import s99_project_functions_bio_embeddings
import argparse
import os

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config")
args = parser.parse_args()
config_filename_tcr = args.config

# Load configs
config_tcr = s99_project_functions_bio_embeddings.load_config(config_filename_tcr)
config_main = s99_project_functions_bio_embeddings.load_config('s97_main_config.yaml')

# Set parameters from config
embedder_name_tcr = config_tcr['default']['embedder_name_tcr']
embedder_index_tcr = config_tcr['default']['embedder_index_tcr']
data_filename = config_main['default']['data_filename']

# Read data
data = pd.read_csv(filepath_or_buffer = os.path.join('../data/raw',
                                                     data_filename),
                   index_col = 'original_index',
                   dtype = {'A1_start': np.ushort,
                            'A1_end': np.ushort,
                            'A2_start': np.ushort,
                            'A2_end': np.ushort,
                            'A3_start': np.ushort,
                            'A3_end': np.ushort,
                            'B1_start': np.ushort,
                            'B1_end': np.ushort,
                            'B2_start': np.ushort,
                            'B2_end': np.ushort,
                            'B3_start': np.ushort,
                            'B3_end': np.ushort},
                   usecols = ['TRA_aa',
                              'TRB_aa',
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
                              'B3_end',
                              'binder',
                              'original_index'])

# Get the embedder
embedder = s99_project_functions_bio_embeddings.get_bio_embedder(name = embedder_name_tcr)

# Do the embedding for positive binders - all unique TCRs
data = (data
        .query('binder == 1')
        .assign(tra_aa_encoded = lambda x: x.TRA_aa.map(embedder.embed),
                trb_aa_encoded = lambda x: x.TRB_aa.map(embedder.embed)))

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

    data[cdr_name] = data.apply(func = lambda x: x[tcr_name][x[cdr_start_name]:x[cdr_end_name]],
                                axis = 1)

data = (data
        .filter(items = cdr_name_tuple))

# Save embeddings
data.to_pickle(path = '../data/s01_et{}_embedding.pkl'.format(embedder_index_tcr))
