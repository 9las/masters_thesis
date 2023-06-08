#!/usr/bin/env python
import numpy as np
import pandas as pd
from bio_embeddings.embed import ESM1bEmbedder

# Read data
data = pd.read_csv(filepath_or_buffer = '../data/raw/nettcr_train_swapped_peptide_ls_3_26_peptides_full_tcr_final.csv',
                   dtype = {'A1_start': np.int32,
                            'A1_end': np.int32,
                            'A2_start': np.int32,
                            'A2_end': np.int32,
                            'A3_start': np.int32,
                            'A3_end': np.int32,
                            'B1_start': np.int32,
                            'B1_end': np.int32,
                            'B2_start': np.int32,
                            'B2_end': np.int32,
                            'B3_start': np.int32,
                            'B3_end': np.int32},
                   index_col = 0)

# Choose the embedding model
embedder = ESM1bEmbedder()

# Do the embedding
data = data \
    .query('binder == 1') \
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
                                            axis = 1))
# Save embedding
data[['A1_embedded',
      'A2_embedded',
      'A3_embedded',
      'B1_embedded',
      'B2_embedded',
      'B3_embedded']].to_pickle(path = '../data/embedding.pkl')
