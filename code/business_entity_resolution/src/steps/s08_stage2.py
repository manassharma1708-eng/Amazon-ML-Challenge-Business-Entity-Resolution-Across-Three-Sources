"""Step 8: stage 2 (LightGBM on stage-1 probability + shop context + error-analysis + reverse features)."""
import os, sys, gc, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import WORK, CLEAN, ids_of, fold0_split, add_reverse_features
import stage2

# fold-0 training table
val = pd.read_parquet(f'{WORK}/feats/fold0_preds_v1.parquet')
S = stage2.run_chunked_v2(val, CLEAN, 'tr_s1', ['tr_s2', 'tr_s3'], stage2.name_counts(CLEAN, ['tr_s1']),
                          stage2.name_counts(CLEAN, ['tr_s2', 'tr_s3']), n_chunks=4)
S = S.merge(val[['source1_entity_id', 'candidate_id', 'label']], on=['source1_entity_id', 'candidate_id'])
S.to_parquet(f'{WORK}/feats/fold0_stage2_v2.parquet', index=False)
del val; gc.collect()
S = add_reverse_features(S, pd.read_parquet(f'{WORK}/cands/reverse_top3_train.parquet'),
                         ids_of(['tr_s1']), ids_of(['tr_s2', 'tr_s3']))
FE = [c for c in S.columns if c not in ('source1_entity_id', 'candidate_id', 'label')]
A, B1, B2 = fold0_split()
tr, b1 = S[S['source1_entity_id'].isin(A)], S[S['source1_entity_id'].isin(B1)]
params = dict(objective='binary', metric='auc', learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7)
m = lgb.train(params, lgb.Dataset(tr[FE], tr['label']), num_boost_round=1000,
              valid_sets=[lgb.Dataset(b1[FE], b1['label'])], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
m.save_model(f'{WORK}/models/lgb_stage2_v3.txt')
del S, tr, b1; gc.collect()

# test
s_ids, p_ids = ids_of(['te_s1']), ids_of(['te_s2', 'te_s3'])
rev = pd.read_parquet(f'{WORK}/cands/reverse_top3_test.parquet')
s1c, pc = stage2.name_counts(CLEAN, ['te_s1']), stage2.name_counts(CLEAN, ['te_s2', 'te_s3'])
preds = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f'{WORK}/test_preds/part_*.parquet'))], ignore_index=True)
OUTD = f'{WORK}/test_preds_s2v3'; os.makedirs(OUTD, exist_ok=True)
for i, chunk in enumerate(np.array_split(preds['source1_entity_id'].unique(), 8)):
    path = f'{OUTD}/part_{i:02d}.parquet'
    if os.path.exists(path):
        continue
    p = preds[preds['source1_entity_id'].isin(set(chunk))]
    bf = stage2.load_fields(CLEAN, ['te_s2', 'te_s3'], p['candidate_id'])
    af = stage2.load_s1_fields(CLEAN, 'te_s1', p['source1_entity_id'])
    ids, F = stage2.stage2_features(p, bf)
    X = add_reverse_features(pd.concat([ids, F, stage2.extra_features(ids, af, bf, s1c, pc)], axis=1), rev, s_ids, p_ids)
    out = X[['source1_entity_id', 'candidate_id']].copy()
    out['prob'] = m.predict(X[m.feature_name()]).astype('float32')
    out.to_parquet(path, index=False)
    print(f'test chunk {i+1}/8'); del p, bf, af, ids, F, X, out; gc.collect()
