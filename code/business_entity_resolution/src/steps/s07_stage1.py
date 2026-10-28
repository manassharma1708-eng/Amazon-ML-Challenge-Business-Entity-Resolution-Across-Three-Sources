"""Step 7: 40 pair features for the top-10 candidates, LightGBM stage 1 (trained on fold 1), test probabilities."""
import os, sys, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import WORK, CLEAN, TRAIN, TEST, N_DOCS
import blocking, features, inference

os.makedirs(f'{WORK}/feats', exist_ok=True); os.makedirs(f'{WORK}/models', exist_ok=True)
name_df, addr_df = blocking.word_frequencies(CLEAN, TRAIN)
for split in ['fold1', 'fold0']:
    p = pd.read_parquet(f'{WORK}/cands/{split}_top30.parquet')
    p = p[p['rank'] < 10].reset_index(drop=True)
    left = features.load_clean_subset(CLEAN, ['tr_s1'], p['source1_entity_id'])
    right = features.load_clean_subset(CLEAN, ['tr_s2', 'tr_s3'], p['candidate_id'])
    df = features.attach(p, left, right); del left, right; gc.collect()
    F = features.compute_features(df, name_df, addr_df, N_DOCS['train'])
    pd.concat([p[['source1_entity_id', 'candidate_id', 'label']], F], axis=1) \
      .to_parquet(f'{WORK}/feats/{split}_top10.parquet', index=False)
    print(split, 'features done'); del p, df, F; gc.collect()

tr = pd.read_parquet(f'{WORK}/feats/fold1_top10.parquet')
FEATS = [c for c in tr.columns if c not in ('source1_entity_id', 'candidate_id', 'label')]
shops = tr['source1_entity_id'].unique()
inner = set(np.random.default_rng(0).choice(shops, size=len(shops) // 7, replace=False))
is_inner = tr['source1_entity_id'].isin(inner).to_numpy()
params = dict(objective='binary', metric='auc', learning_rate=0.08, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42)
model = lgb.train(params, lgb.Dataset(tr.loc[~is_inner, FEATS], tr.loc[~is_inner, 'label']), num_boost_round=1500,
                  valid_sets=[lgb.Dataset(tr.loc[is_inner, FEATS], tr.loc[is_inner, 'label'])],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
model.save_model(f'{WORK}/models/lgb_v1.txt')
del tr; gc.collect()
va = pd.read_parquet(f'{WORK}/feats/fold0_top10.parquet')
va['prob'] = model.predict(va[FEATS], num_iteration=model.best_iteration).astype('float32')
va[['source1_entity_id', 'candidate_id', 'label', 'prob']].to_parquet(f'{WORK}/feats/fold0_preds_v1.parquet', index=False)
del va; gc.collect()

name_df, addr_df = blocking.word_frequencies(CLEAN, TEST)
cands = pd.read_parquet(f'{WORK}/cands/test_top30.parquet',
                        columns=['source1_entity_id', 'candidate_id', 'score', 'n_keys', 'rank'], filters=[('rank', '<', 10)])
os.makedirs(f'{WORK}/test_preds', exist_ok=True)
inference.score_candidates(cands, CLEAN, 'te_s1', ['te_s2', 'te_s3'], name_df, addr_df, N_DOCS['test'],
                           model, model.feature_name(), n_chunks=8, out_dir=f'{WORK}/test_preds')
print('stage 1 done')
