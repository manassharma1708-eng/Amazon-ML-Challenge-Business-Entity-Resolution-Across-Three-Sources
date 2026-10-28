"""Step 12 (GPU): MiniLM-L12 cross-encoder on more data (fold-0 A + fold 1); stage 4 stacks it on the v7 probability."""
import os, sys, gc, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import WORK, TRAIN, TEST, CE_BAND, texts_for, fold0_split
import ce_utils

LO, HI = CE_BAND
IDS = ['source1_entity_id', 'candidate_id']
A, B1, B2 = fold0_split()


def s4_feats(df):
    g = df.groupby('source1_entity_id')
    X = pd.DataFrame(index=df.index)
    X['p1'], X['p7'], X['ce3'] = df['p1'], df['p7'], df['ce3']
    X['p7_rank'] = g['p7'].rank(ascending=False, method='first')
    X['p7_gap'] = df['p7'] - g['p7'].transform('max')
    X['p7_share'] = df['p7'] / g['p7'].transform('sum').clip(lower=1e-6)
    X['ce3_rank'] = g['ce3'].rank(ascending=False, method='first')
    X['ce3_gap'] = df['ce3'] - g['ce3'].transform('max')
    return X.astype('float32')


# ---- training pairs: uncertain fold-0 A pairs + uncertain fold-1 pairs ----
S = pd.read_parquet(f'{WORK}/feats/fold0_stage2_v2.parquet', columns=IDS + ['label', 'prob'])
p1_fold0 = S[IDS + ['prob']].rename(columns={'prob': 'p1'})
trA = S[S['source1_entity_id'].isin(A) & S['prob'].between(LO, HI)][IDS + ['label']]
del S
m1 = lgb.Booster(model_file=f'{WORK}/models/lgb_v1.txt')
f1 = pd.read_parquet(f'{WORK}/feats/fold1_top10.parquet')
p = m1.predict(f1[m1.feature_name()])                     # only used to select the uncertain band
trF1 = f1.loc[(p >= 0.005) & (p <= 0.995), IDS + ['label']]
del f1, p; gc.collect()
trce = pd.concat([trA, trF1], ignore_index=True).sample(frac=1, random_state=0).head(1_200_000).reset_index(drop=True)
tx = texts_for(np.concatenate([trce['source1_entity_id'], trce['candidate_id']]), TRAIN)
tok, ce = ce_utils.train_ce('cross-encoder/ms-marco-MiniLM-L-12-v2', [tx[i] for i in trce['source1_entity_id']],
                            [tx[i] for i in trce['candidate_id']], trce['label'], f'{WORK}/models/cross_encoder_v3',
                            epochs=1, lr=3e-5)
del tx, trce; gc.collect()

# ---- stage 4: trained on B2a, reported on B2b ----
ev = pd.read_parquet(f'{WORK}/feats/fold0_B_v7probs.parquet').rename(columns={'prob': 'p7'})
ev = ev[ev['source1_entity_id'].isin(B2)].merge(p1_fold0, on=IDS, how='left').reset_index(drop=True)
u = ev['p1'].between(LO, HI).to_numpy()
tx = texts_for(np.concatenate([ev.loc[u, 'source1_entity_id'], ev.loc[u, 'candidate_id']]), TRAIN)
ev['ce3'] = np.nan
ev.loc[u, 'ce3'] = ce_utils.score(tok, ce, [tx[i] for i in ev.loc[u, 'source1_entity_id']], [tx[i] for i in ev.loc[u, 'candidate_id']])
del tx; gc.collect()
X = s4_feats(ev)
b2s = np.array(sorted(B2)); np.random.default_rng(4).shuffle(b2s)
B2a = set(b2s[:len(b2s) // 2])
b2a_s = np.array(sorted(B2a)); np.random.default_rng(5).shuffle(b2a_s)
tr_ids = set(b2a_s[:int(.8 * len(b2a_s))])
m_tr = ev['source1_entity_id'].isin(tr_ids).to_numpy()
m_va = ev['source1_entity_id'].isin(B2a).to_numpy() & ~m_tr
params = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=13)
m4 = lgb.train(params, lgb.Dataset(X[m_tr], ev.loc[m_tr, 'label']), num_boost_round=1000,
               valid_sets=[lgb.Dataset(X[m_va], ev.loc[m_va, 'label'])], callbacks=[lgb.early_stopping(50)])
m4.save_model(f'{WORK}/models/lgb_stage4_ce3.txt')
del ev, X; gc.collect()

# ---- test ----
parts1 = sorted(glob.glob(f'{WORK}/test_preds/part_*.parquet'))
parts7 = sorted(glob.glob(f'{WORK}/test_preds_s3v2/part_*.parquet'))
need = set()
for f in parts1:
    d = pd.read_parquet(f); d = d[d['prob'].between(LO, HI)]
    need.update(d['source1_entity_id']); need.update(d['candidate_id'])
tx = texts_for(need, TEST); del need
OUTD = f'{WORK}/test_preds_s4'; os.makedirs(OUTD, exist_ok=True)
for i, (f1, f7) in enumerate(zip(parts1, parts7)):
    path = f'{OUTD}/part_{i:02d}.parquet'
    if os.path.exists(path):
        continue
    d = pd.read_parquet(f1).rename(columns={'prob': 'p1'}).merge(
        pd.read_parquet(f7, columns=IDS + ['prob']).rename(columns={'prob': 'p7'}), on=IDS)
    d['ce3'] = np.nan
    a, b = d['source1_entity_id'].map(tx), d['candidate_id'].map(tx)
    ok = d['p1'].between(LO, HI).to_numpy() & a.notna().to_numpy() & b.notna().to_numpy()
    d.loc[ok, 'ce3'] = ce_utils.score(tok, ce, a[ok].tolist(), b[ok].tolist())
    d['prob'] = m4.predict(s4_feats(d), num_iteration=m4.best_iteration).astype('float32')
    d[IDS + ['ce3', 'prob']].to_parquet(path, index=False)
    print(f'test chunk {i+1}/8'); del d; gc.collect()
