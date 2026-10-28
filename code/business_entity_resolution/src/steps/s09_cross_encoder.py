"""Step 9 (GPU): fine-tune the MiniLM-L6 cross-encoder on uncertain pairs, stage 3 on B1, test probabilities."""
import os, sys, gc, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import WORK, TRAIN, TEST, CE_BAND, ids_of, texts_for, fold0_split, add_reverse_features
import ce_utils

LO, HI = CE_BAND
A, B1, B2 = fold0_split()
S = add_reverse_features(pd.read_parquet(f'{WORK}/feats/fold0_stage2_v2.parquet'),
                         pd.read_parquet(f'{WORK}/cands/reverse_top3_train.parquet'), ids_of(['tr_s1']), ids_of(['tr_s2', 'tr_s3']))
S['p1'] = S['prob']
S['unc'] = S['p1'].between(LO, HI)
trce = S[S['source1_entity_id'].isin(A) & S['unc']][['source1_entity_id', 'candidate_id', 'label']]
trce = trce.sample(min(len(trce), 800_000), random_state=0).reset_index(drop=True)
m2 = lgb.Booster(model_file=f'{WORK}/models/lgb_stage2_v3.txt')
ev = S[S['source1_entity_id'].isin(B1 | B2)].reset_index(drop=True)
del S; gc.collect()
ev['p3'] = m2.predict(ev[m2.feature_name()]).astype('float32')
ev = ev[['source1_entity_id', 'candidate_id', 'label', 'p1', 'p3', 'unc']]

u = ev['unc'].to_numpy()
tx = texts_for(np.concatenate([trce['source1_entity_id'], trce['candidate_id'],
                               ev.loc[u, 'source1_entity_id'], ev.loc[u, 'candidate_id']]), TRAIN)
tok, ce = ce_utils.train_ce('cross-encoder/ms-marco-MiniLM-L-6-v2', [tx[i] for i in trce['source1_entity_id']],
                            [tx[i] for i in trce['candidate_id']], trce['label'], f'{WORK}/models/cross_encoder_v2', epochs=2)
ev['ce'] = np.nan
ev.loc[u, 'ce'] = ce_utils.score(tok, ce, [tx[i] for i in ev.loc[u, 'source1_entity_id']], [tx[i] for i in ev.loc[u, 'candidate_id']])
del tx; gc.collect()
X = ce_utils.stage3_feats(ev)
b1s = np.array(sorted(B1)); np.random.default_rng(2).shuffle(b1s)
b1_tr = set(b1s[:int(.8 * len(b1s))])
m_tr = ev['source1_entity_id'].isin(b1_tr).to_numpy()
m_va = ev['source1_entity_id'].isin(B1).to_numpy() & ~m_tr
params = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=11)
m3 = lgb.train(params, lgb.Dataset(X[m_tr], ev.loc[m_tr, 'label']), num_boost_round=1000,
               valid_sets=[lgb.Dataset(X[m_va], ev.loc[m_va, 'label'])], callbacks=[lgb.early_stopping(50)])
m3.save_model(f'{WORK}/models/lgb_stage3_ce2.txt')
ev['prob'] = m3.predict(X, num_iteration=m3.best_iteration).astype('float32')
ev[['source1_entity_id', 'candidate_id', 'label', 'prob']].to_parquet(f'{WORK}/feats/fold0_B_v7probs.parquet', index=False)
del ev, X; gc.collect()

# test
parts1 = sorted(glob.glob(f'{WORK}/test_preds/part_*.parquet'))
parts3 = sorted(glob.glob(f'{WORK}/test_preds_s2v3/part_*.parquet'))
need = set()
for f in parts1:
    d = pd.read_parquet(f); d = d[d['prob'].between(LO, HI)]
    need.update(d['source1_entity_id']); need.update(d['candidate_id'])
tx = texts_for(need, TEST); del need
OUTD = f'{WORK}/test_preds_s3v2'; os.makedirs(OUTD, exist_ok=True)
for i, (f1, f3) in enumerate(zip(parts1, parts3)):
    path = f'{OUTD}/part_{i:02d}.parquet'
    if os.path.exists(path):
        continue
    d = pd.read_parquet(f1).rename(columns={'prob': 'p1'}).merge(
        pd.read_parquet(f3).rename(columns={'prob': 'p3'}), on=['source1_entity_id', 'candidate_id'])
    d['ce'] = np.nan
    uu = d['p1'].between(LO, HI).to_numpy()
    d.loc[uu, 'ce'] = ce_utils.score(tok, ce, [tx[x] for x in d.loc[uu, 'source1_entity_id']], [tx[x] for x in d.loc[uu, 'candidate_id']])
    d['prob'] = m3.predict(ce_utils.stage3_feats(d), num_iteration=m3.best_iteration).astype('float32')
    d[['source1_entity_id', 'candidate_id', 'ce', 'prob']].to_parquet(path, index=False)
    print(f'test chunk {i+1}/8'); del d; gc.collect()
