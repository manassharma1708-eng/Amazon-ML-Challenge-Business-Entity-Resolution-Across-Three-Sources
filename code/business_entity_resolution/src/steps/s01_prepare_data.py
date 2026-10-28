"""Step 1: read the raw TSV files (tab-separated), store them as Parquet, and build the true-pair table."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from common import DATA, WORK, RAW_FILES

os.makedirs(WORK, exist_ok=True)
load = lambda p: pd.read_csv(p, sep='\t', dtype=str, keep_default_na=False)
for name, rel in RAW_FILES.items():
    load(f'{DATA}/{rel}').astype('string[pyarrow]').to_parquet(f'{WORK}/{name}.parquet', index=False)
    print('saved', name)
gt = load(f'{DATA}/train/train_ground_truth.tsv')
gt.to_parquet(f'{WORK}/gt.parquet', index=False)
m = gt['matched_entity_ids']
gt['n'] = (m.str.count(',') + 1).where(m != '', 0)
pairs = gt[gt['n'] > 0][['source1_entity_id', 'matched_entity_ids']].copy()
pairs['match_id'] = pairs['matched_entity_ids'].str.split(',')
pairs = pairs.explode('match_id')[['source1_entity_id', 'match_id']]
pairs['src'] = pairs['match_id'].str[:2]
pairs.to_parquet(f'{WORK}/train_pairs.parquet', index=False)
print(f'true pairs: {len(pairs):,}')
