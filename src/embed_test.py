#!/usr/bin/env python
import pandas as pd
import time
from bio_embeddings.embed import ESM1bEmbedder
data = pd.read_csv(filepath_or_buffer = '../data/raw/nettcr_train_swapped_peptide_ls_3_26_peptides_full_tcr_final.csv',
                   index_col = 0)

embedder = ESM1bEmbedder()
for i in range(10):
    start_time = time.time()
    embedding = embedder.embed(data.TRA_aa[i])
    print(time.time()-start_time)
    print(embedding.shape)
