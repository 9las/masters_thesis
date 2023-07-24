#!/usr/bin/env python
import numpy as np
import pandas as pd
import s99_project_functions_bio_embeddings
import argparse

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config")
args = parser.parse_args()
config_filename = args.config

# Load config
config = s99_project_functions_bio_embeddings.load_config(config_filename)

# Set parameters from config
embedder_name_peptide = config['default']['embedder_name_peptide']

# Read data
data = pd.read_csv(filepath_or_buffer = '../data/raw/nettcr_train_swapped_peptide_ls_3_26_peptides_full_tcr_final.csv',
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
data.to_pickle(path = '../data/s01_embedding_peptide_{}.pkl'.format(embedder_name_peptide))
