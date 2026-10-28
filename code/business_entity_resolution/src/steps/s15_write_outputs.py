"""Step 15 (final): stage-4 decisions + union-1, union-2 and embedding extras -> the two output TSVs (+ validator)."""
import os, sys, glob, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from common import WORK, CLEAN, OUT, DATA
import decide, inference

IDS = ['source1_entity_id', 'candidate_id']
os.makedirs(OUT, exist_ok=True)
all_ids = pd.read_parquet(f'{CLEAN}/te_s1', columns=['entity_id'])['entity_id'].astype(str).tolist()
s4 = pd.concat([pd.read_parquet(p, columns=IDS + ['prob']) for p in sorted(glob.glob(f'{WORK}/test_preds_s4/part_*.parquet'))], ignore_index=True)
sel = decide.expected_f05(decide.one_to_one(s4), 0.1).sort_values(['source1_entity_id', 'prob'], ascending=[True, False])
u1 = pd.read_parquet(f'{WORK}/test_union_extras.parquet')
u2 = pd.read_parquet(f'{WORK}/test_union2_extras.parquet')
ue = pd.read_parquet(f'{WORK}/test_emb_extras.parquet')
acc1 = u1.loc[u1['accept'] & ~u1['candidate_id'].isin(set(sel['candidate_id'])), IDS]
matches = pd.concat([sel[IDS], acc1, u2.loc[u2['accept'], IDS], ue.loc[ue['accept'], IDS]])
inference.write_lists(all_ids, matches, 'matched_entity_ids', f'{OUT}/matching_results.tsv')

top10 = pd.read_parquet(f'{WORK}/cands/test_top30.parquet', columns=IDS + ['rank'], filters=[('rank', '<', 10)]).sort_values(['source1_entity_id', 'rank'])
cands = pd.concat([top10[IDS], u1[IDS], u2[IDS], ue[IDS]])            # every pair the models ran inference on
inference.write_lists(all_ids, cands, 'candidate_entity_ids', f'{OUT}/candidate_pairs.tsv')
print(f'matches/shop {len(matches)/len(all_ids):.3f} | candidates/shop {len(cands)/len(all_ids):.2f}')

validator = os.path.join(os.path.dirname(DATA), 'utils', 'validate_submission.py')
if os.path.exists(validator):
    subprocess.run([sys.executable, validator, '--matching', f'{OUT}/matching_results.tsv',
                    '--candidate', f'{OUT}/candidate_pairs.tsv', '--test-dir', f'{DATA}/test'], check=False)
