#!/usr/bin/env python
import numpy as np
import pandas as pd
from bio_embeddings.embed import ESM1bEmbedder
import s99_project_functions

# Read data
data = pd.read_csv(filepath_or_buffer = '../data/raw/nettcr_train_swapped_peptide_ls_3_26_peptides_full_tcr_final.csv',
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
                              'binder'])

# Choose the embedding model
embedder = ESM1bEmbedder()

# Do the embedding on full TCR chains
data = (data
    .query('binder == 1')
    .assign(TRA_aa_embedded = lambda x: x.TRA_aa.map(embedder.embed),
            TRB_aa_embedded = lambda x: x.TRB_aa.map(embedder.embed)))

# Initialize
cdr_embedded_dict = dict()
cdr_name_tuple = ('a1', 'a2', 'a3', 'b1', 'b2', 'b3')
chain_name_tuple = ('TRA_aa_embedded',
                    'TRA_aa_embedded',
                    'TRA_aa_embedded',
                    'TRB_aa_embedded',
                    'TRB_aa_embedded',
                    'TRB_aa_embedded')
start_name_tuple = ('A1_start',
                    'A2_start',
                    'A3_start',
                    'B1_start',
                    'B2_start',
                    'B3_start')
end_name_tuple = ('A1_end',
                  'A2_end',
                  'A3_end',
                  'B1_end',
                  'B2_end',
                  'B3_end')

# Extract and pad embedded CDR sequences
for i in range(len(cdr_name_tuple)):
    cdr_name = cdr_name_tuple[i]
    chain_name = chain_name_tuple[i]
    start_name = start_name_tuple[i]
    end_name = end_name_tuple[i]
    # Extract
    cdr_embedded_dict[cdr_name] = data.apply(func = lambda x: x[chain_name][x[start_name]:x[end_name]],
                                             axis = 1).tolist()

    # Pad
    cdr_embedded_dict[cdr_name] = (s99_project_functions
                                   .pad_sequences(sequence_array = cdr_embedded_dict[cdr_name],
                                   padding_value = 0))

# Save embedding
np.savez_compressed(file = '../data/embedding.npz',
                    **cdr_embedded_dict)
