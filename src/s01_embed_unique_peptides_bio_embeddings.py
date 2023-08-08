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
config_filename_peptide = args.config

# Load configs
config_peptide = s99_project_functions_bio_embeddings.load_config(config_filename_peptide)
config_main = s99_project_functions_bio_embeddings.load_config('s97_main_config.yaml')

# Set parameters from configs
embedder_name_peptide = config_peptide['default']['embedder_name_peptide']
embedder_index_peptide = config_peptide['default']['embedder_index_peptide']
data_filename = config_main['default']['data_filename']

# Read data
data = pd.read_csv(filepath_or_buffer = os.path.join('../data/raw',
                                                     data_filename),
                   usecols = ['peptide'])

# Get the embedder
embedder = s99_project_functions_bio_embeddings.get_bio_embedder(name = embedder_name_peptide)

# Do the embedding for all unique peptides
data = (data
        .groupby('peptide')
        .agg(count = ('peptide',
                      'count'))
        .assign(peptide_encoded = lambda x: x.index.map(embedder.embed)))

# Save embeddings
data.to_pickle(path = '../data/s01_ep{}_embedding.pkl'.format(embedder_index_peptide))
