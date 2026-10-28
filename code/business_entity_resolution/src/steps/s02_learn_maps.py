"""Step 2: learn the transliteration word map and regional-script state aliases from TRAINING pairs (India)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from common import WORK, TRAIN
import learn_maps

tables = [pd.read_parquet(f'{WORK}/{s}.parquet', dtype_backend='pyarrow') for s in TRAIN]
lookup = pd.concat(tables).set_index('entity_id')
del tables
pairs = pd.read_parquet(f'{WORK}/train_pairs.parquet').astype(str)
country = lookup['country']
pairs['country'] = country.loc[pairs['source1_entity_id']].astype(str).values
pi = pairs[pairs['country'] == 'India']
col = lambda c, ids: lookup[c].loc[ids].astype(str).tolist()
n1, n2 = col('business_name', pi['source1_entity_id']), col('business_name', pi['match_id'])
a1, a2 = col('business_address', pi['source1_entity_id']), col('business_address', pi['match_id'])
word_map = learn_maps.learn_word_map(n1, n2)
state_map = learn_maps.learn_state_map(a1, a2, 'India')     # addresses.STATE_MAPS holds no learned aliases here
learn_maps.save_maps(f'{WORK}/maps.json', word_map, {'India': state_map})
print(f'word mappings {len(word_map)}, state aliases {len(state_map)}')
