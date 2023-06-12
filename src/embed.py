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

# Do the embedding
data = (data
    .query('binder == 1')
    .assign(TRA_aa_embedded = lambda x: x.TRA_aa.map(embedder.embed),
            TRB_aa_embedded = lambda x: x.TRB_aa.map(embedder.embed),
            A1_embedded = lambda x: x.apply(func = lambda y: y.TRA_aa_embedded[y.A1_start:y.A1_end],
                                            axis = 1),
            A2_embedded = lambda x: x.apply(func = lambda y: y.TRA_aa_embedded[y.A2_start:y.A2_end],
                                            axis = 1),
            A3_embedded = lambda x: x.apply(func = lambda y: y.TRA_aa_embedded[y.A3_start:y.A3_end],
                                            axis = 1),
            B1_embedded = lambda x: x.apply(func = lambda y: y.TRB_aa_embedded[y.B1_start:y.B1_end],
                                            axis = 1),
            B2_embedded = lambda x: x.apply(func = lambda y: y.TRB_aa_embedded[y.B2_start:y.B2_end],
                                            axis = 1),
            B3_embedded = lambda x: x.apply(func = lambda y: y.TRB_aa_embedded[y.B3_start:y.B3_end],
                                            axis = 1)))

# Pad embedded sequences
a1_padded = s99_project_functions.pad_sequences(sequence_array = data.A1_embedded.tolist(),
                                                padding_value = -5)
a2_padded = s99_project_functions.pad_sequences(sequence_array = data.A2_embedded.tolist(),
                                                padding_value = -5)
a3_padded = s99_project_functions.pad_sequences(sequence_array = data.A3_embedded.tolist(),
                                                padding_value = -5)
b1_padded = s99_project_functions.pad_sequences(sequence_array = data.B1_embedded.tolist(),
                                                padding_value = -5)
b2_padded = s99_project_functions.pad_sequences(sequence_array = data.B2_embedded.tolist(),
                                                padding_value = -5)
b3_padded = s99_project_functions.pad_sequences(sequence_array = data.B3_embedded.tolist(),
                                                padding_value = -5)

# Save embedding
np.savez_compressed(file = '../data/embedding.npz',
                    a1_padded = a1_padded,
                    a2_padded = a2_padded,
                    a3_padded = a3_padded,
                    b1_padded = b1_padded,
                    b2_padded = b2_padded,
                    b3_padded = b3_padded)
