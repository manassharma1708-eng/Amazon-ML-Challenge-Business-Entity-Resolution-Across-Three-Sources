"""Step 14 (GPU, ~30 GB RAM): embedding search - each still-unmatched record's 3 nearest S1 shops; a cheap pre-ranker,
then the L12 cross-encoder and add-on 3 decide. Final settings EMB_T and EXCLUDE were chosen on the public leaderboard."""
import os, sys, gc, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
from anyascii import anyascii
from common import WORK, CLEAN, TRAIN, TEST, ids_of, texts_for, fold0_split
import ce_utils, decide, emb_utils

IDS = ['source1_entity_id', 'candidate_id']
EMB_T = 0.9                 # acceptance threshold (B1 optimum was 0.6; 0.9 scored best on the public leaderboard)
EXCLUDE = {'France'}        # no French training examples; French extras lowered the public score at every threshold
FPRE = ['emb_sim', 'emb_rank', 'emb_top1', 'emb_gap', 'emb_margin', 'shop_nq', 's1_rank', 's1_score', 'shop_best', 'n_acc']
FEX3 = FPRE + ['ce']
M = 20_000_000


def raw_texts(sources, keep=None):
    out = []
    for s in sources:
        r = pd.read_parquet(f'{WORK}/{s}.parquet', dtype_backend='pyarrow')
        if keep is not None:
            r = r[r['entity_id'].isin(keep)]
        r = r.astype(str)
        t = [f'{n} | {a}' for n, a in zip(r['business_name'], r['business_address'])]
        out.append(pd.DataFrame({'entity_id': r['entity_id'].values, 'country': r['country'].values,
                                 'text': [x if x.isascii() else anyascii(x) for x in t]}))
    return pd.concat(out, ignore_index=True)


def neighbours(s1_src, q_src, q_keep, etok, emod):
    s1 = raw_texts([s1_src]); S = emb_utils.embed(etok, emod, s1['text'].tolist())
    q = raw_texts(q_src, q_keep); Q = emb_utils.embed(etok, emod, q['text'].tolist())
    nn = emb_utils.search(S, s1['country'].to_numpy(), s1['entity_id'].to_numpy(), Q, q['country'].to_numpy(), q['entity_id'].to_numpy())
    del S, Q; torch.cuda.empty_cache()
    return emb_utils.emb_feats(nn)


def finish(nn, s_srcs, p_srcs, top30, rev_path, probs, decided):
    s_index, p_index = pd.Index(ids_of(s_srcs)), pd.Index(ids_of(p_srcs))
    code = s_index.get_indexer(nn['source1_entity_id']).astype(np.int64) * M + p_index.get_indexer(nn['candidate_id'])
    t10 = top30[top30['rank'] < 10]
    seen = s_index.get_indexer(t10['source1_entity_id']).astype(np.int64) * M + p_index.get_indexer(t10['candidate_id'])
    rv = pd.read_parquet(rev_path, columns=['q_row', 'p_row'])
    seen = np.concatenate([seen, rv['p_row'].to_numpy().astype(np.int64) * M + rv['q_row'].to_numpy()])
    nn = nn[~np.isin(code, seen)].reset_index(drop=True)                  # pairs never considered before
    nn = nn.merge(top30[top30['rank'] >= 10].rename(columns={'rank': 's1_rank', 'score': 's1_score'}), how='left', on=IDS)
    nn['shop_best'] = nn['source1_entity_id'].map(probs.groupby('source1_entity_id')['prob'].max()).fillna(0)
    nn['n_acc'] = nn['source1_entity_id'].map(decided.groupby('source1_entity_id').size()).fillna(0)
    return nn


def add_ce(x, sources, tok, ce):
    tx = texts_for(np.concatenate([x['source1_entity_id'], x['candidate_id']]), sources)
    a, b = x['source1_entity_id'].map(tx), x['candidate_id'].map(tx)
    ok = (a.notna() & b.notna()).to_numpy()
    x = x[ok].reset_index(drop=True)
    x['ce'] = ce_utils.score(tok, ce, a[ok].tolist(), b[ok].tolist())
    return x


etok, emod = emb_utils.load()
tok, ce = ce_utils.load_ce(f'{WORK}/models/cross_encoder_v3')

# ================= validation (B1 trains, B2 reports) =================
A, B1, B2 = fold0_split(); B = B1 | B2
ev = pd.read_parquet(f'{WORK}/feats/fold0_B_v7probs.parquet')
base = pd.concat([decide.expected_f05(decide.one_to_one(ev), 0.1)[IDS],
                  pd.read_parquet(f'{WORK}/feats/fold0_B_union1_accepted.parquet')], ignore_index=True)
pairs = pd.read_parquet(f'{WORK}/train_pairs.parquet')[['source1_entity_id', 'match_id']].astype(str).rename(columns={'match_id': 'candidate_id'})
owner = pd.Series(pairs['source1_entity_id'].values, index=pairs['candidate_id'].values)
p_ids = pd.Series(ids_of(TRAIN[1:]))
own = p_ids.map(owner)
# queries: unmatched records; records owned by non-B shops are left out (on test their own shop claims them)
val_q = pd.Index(p_ids[~p_ids.isin(set(base['candidate_id'])) & (own.isna() | own.isin(B))].values)
del owner, p_ids, own; gc.collect()
nv = neighbours('tr_s1', TRAIN[1:], val_q, etok, emod)
nv = nv[nv['source1_entity_id'].isin(B)].reset_index(drop=True)
c30 = pd.read_parquet(f'{WORK}/cands/fold0_top30.parquet', columns=IDS + ['rank', 'score'])
nv = finish(nv, TRAIN[:1], TRAIN[1:], c30, f'{WORK}/cands/reverse_top3_train.parquet', ev, base)
nv = nv.merge(pairs.assign(label=1), how='left', on=IDS)
nv['label'] = nv['label'].fillna(0).astype('int8')
thr = float(nv.loc[nv['label'] == 1, 'emb_sim'].quantile(0.01))
nv = nv[nv['emb_sim'] >= thr].reset_index(drop=True)
del c30, pairs; gc.collect()

b1s = np.array(sorted(B1)); np.random.default_rng(7).shuffle(b1s)
b1_tr = set(b1s[:int(.8 * len(b1s))])
t_mask = nv['source1_entity_id'].isin(b1_tr).to_numpy()
v_mask = nv['source1_entity_id'].isin(B1).to_numpy() & ~t_mask
pre = lgb.train(dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
                     feature_fraction=0.9, verbose=-1, seed=9),
                lgb.Dataset(nv.loc[t_mask, FPRE], nv.loc[t_mask, 'label']), num_boost_round=1000,
                valid_sets=[lgb.Dataset(nv.loc[v_mask, FPRE], nv.loc[v_mask, 'label'])], callbacks=[lgb.early_stopping(50)])
nv['p_pre'] = pre.predict(nv[FPRE], num_iteration=pre.best_iteration)
cut = float(nv.loc[v_mask & (nv['label'] == 1).to_numpy(), 'p_pre'].quantile(0.03))   # keeps 97% of true copies
nv = add_ce(nv[nv['p_pre'] >= cut], TRAIN, tok, ce)
t_mask = nv['source1_entity_id'].isin(b1_tr).to_numpy()
v_mask = nv['source1_entity_id'].isin(B1).to_numpy() & ~t_mask
addon3 = lgb.train(dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=15, min_data_in_leaf=50,
                        feature_fraction=0.9, verbose=-1, seed=10),
                   lgb.Dataset(nv.loc[t_mask, FEX3], nv.loc[t_mask, 'label']), num_boost_round=1000,
                   valid_sets=[lgb.Dataset(nv.loc[v_mask, FEX3], nv.loc[v_mask, 'label'])], callbacks=[lgb.early_stopping(50)])
pre.save_model(f'{WORK}/models/lgb_emb_preranker.txt'); addon3.save_model(f'{WORK}/models/lgb_union_addon3.txt')
del nv, ev, base; gc.collect()

# ================= test =================
s4 = pd.concat([pd.read_parquet(p, columns=IDS + ['prob']) for p in sorted(glob.glob(f'{WORK}/test_preds_s4/part_*.parquet'))], ignore_index=True)
sel9 = decide.expected_f05(decide.one_to_one(s4), 0.1)[IDS]
del s4; gc.collect()
u1 = pd.read_parquet(f'{WORK}/test_union_extras.parquet')
u2 = pd.read_parquet(f'{WORK}/test_union2_extras.parquet')
v10 = pd.concat([sel9, u1.loc[u1['accept'] & ~u1['candidate_id'].isin(set(sel9['candidate_id'])), IDS],
                 u2.loc[u2['accept'], IDS]], ignore_index=True)
p_te = pd.Series(ids_of(TEST[1:]))
test_q = pd.Index(p_te[~p_te.isin(set(v10['candidate_id']))].values)
nt = neighbours('te_s1', TEST[1:], test_q, etok, emod)
nt = nt[nt['emb_sim'] >= thr].reset_index(drop=True)
t30 = pd.read_parquet(f'{WORK}/cands/test_top30.parquet', columns=IDS + ['rank', 'score'])
s7 = pd.concat([pd.read_parquet(p, columns=['source1_entity_id', 'prob']) for p in sorted(glob.glob(f'{WORK}/test_preds_s3v2/part_*.parquet'))])
nt = finish(nt, TEST[:1], TEST[1:], t30, f'{WORK}/cands/reverse_top3_test.parquet', s7, v10)
del t30, s7; gc.collect()
nt = nt[pre.predict(nt[FPRE], num_iteration=pre.best_iteration) >= cut].reset_index(drop=True)
nt = add_ce(nt, TEST, tok, ce)
nt['p'] = addon3.predict(nt[FEX3], num_iteration=addon3.best_iteration)
nt['country'] = nt['source1_entity_id'].map(pd.read_parquet(f'{CLEAN}/te_s1', columns=['entity_id', 'country'])
                                             .astype(str).set_index('entity_id')['country'])
acc = nt[(nt['p'] >= EMB_T) & ~nt['country'].isin(EXCLUDE)].sort_values('p', ascending=False).drop_duplicates('candidate_id')
nt['accept'] = False
nt.loc[acc.index, 'accept'] = True
nt[IDS + ['p', 'country', 'accept']].to_parquet(f'{WORK}/test_emb_extras.parquet', index=False)
print(f'embedding extras scored {len(nt):,}, accepted {nt["accept"].sum():,}')
