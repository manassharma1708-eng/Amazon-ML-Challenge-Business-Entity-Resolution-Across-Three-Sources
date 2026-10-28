"""Step 4: assign every training S1 shop to fold 0-4 (stratified by country, seed 42)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from common import WORK, CLEAN

s1 = pd.read_parquet(f'{CLEAN}/tr_s1', columns=['entity_id', 'country'])
rng = np.random.default_rng(42)
folds = s1.rename(columns={'entity_id': 'source1_entity_id'}).copy()
folds['fold'] = -1
for country, idx in folds.groupby('country').groups.items():
    shuffled = rng.permutation(np.asarray(idx))
    folds.loc[shuffled, 'fold'] = np.arange(len(shuffled)) % 5
folds['fold'] = folds['fold'].astype('int8')
folds.to_parquet(f'{WORK}/folds.parquet', index=False)
print(pd.crosstab(folds['fold'], folds['country']))
