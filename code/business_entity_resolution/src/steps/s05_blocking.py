"""Step 5: compound-key blocking -> top-30 candidates per S1 shop (the pipeline uses the top 10)."""
import os, sys, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from common import WORK, CLEAN, TRAIN, TEST
import blocking

os.makedirs(f'{WORK}/cands', exist_ok=True)
folds = pd.read_parquet(f'{WORK}/folds.parquet')
pairs = pd.read_parquet(f'{WORK}/train_pairs.parquet')[['source1_entity_id', 'match_id']].astype(str) \
          .rename(columns={'match_id': 'candidate_id'})

name_df, addr_df = blocking.word_frequencies(CLEAN, TRAIN)
for fold in [0, 1]:
    ids = set(folds.loc[folds['fold'] == fold, 'source1_entity_id'])
    c, _, _ = blocking.run_blocking(CLEAN, 'tr_s1', ['tr_s2', 'tr_s3'], name_df, addr_df, keep_ids=ids, top_k=30)
    c = c.merge(pairs[pairs['source1_entity_id'].isin(ids)].assign(label=np.int8(1)),
                on=['source1_entity_id', 'candidate_id'], how='left')
    c['label'] = c['label'].fillna(0).astype('int8')
    c.to_parquet(f'{WORK}/cands/fold{fold}_top30.parquet', index=False)
    print(f'fold {fold}: {len(c):,} pairs'); del c; gc.collect()

name_df, addr_df = blocking.word_frequencies(CLEAN, TEST)          # rarity measured on the data being searched
c, _, _ = blocking.run_blocking(CLEAN, 'te_s1', ['te_s2', 'te_s3'], name_df, addr_df, top_k=30)
c.to_parquet(f'{WORK}/cands/test_top30.parquet', index=False)
print(f'test: {len(c):,} pairs')
