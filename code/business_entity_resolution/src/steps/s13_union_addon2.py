"""Step 13 (GPU): union add-on 2 - an unmatched record's reverse #2 / #3 shops, scored by the L12 cross-encoder."""
import os, sys, gc, glob, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import WORK, TRAIN, TEST, ids_of, texts_for, fold0_split
import ce_utils, decide, evaluate

IDS = ['source1_entity_id', 'candidate_id']
FEX2 = ['ce', 'rev_rank', 'rev_score', 'rev_best', 'rev_gap', 's1_rank', 's1_score', 'shop_best', 'n_acc']
tok, ce = ce_utils.load_ce(f'{WORK}/models/cross_encoder_v3')


def rev23(path, s_srcs, p_srcs, s_filter=None):
    s_ids, p_ids = ids_of(s_srcs), ids_of(p_srcs)
    rv = pd.read_parquet(path)
    best = rv[rv['rank'] == 0].set_index('q_row')['score']
    rv = rv[rv['rank'].isin([1, 2])]
    if s_filter is not None:
        keep = np.zeros(len(s_ids), bool); keep[pd.Index(s_ids).get_indexer(list(s_filter))] = True
        rv = rv[keep[rv['p_row'].to_numpy()]]
    out = pd.DataFrame({'source1_entity_id': s_ids[rv['p_row'].to_numpy()], 'candidate_id': p_ids[rv['q_row'].to_numpy()],
                        'rev_rank': rv['rank'].to_numpy().astype('float32'), 'rev_score': rv['score'].to_numpy(),
                        'rev_best': rv['q_row'].map(best).to_numpy()})
    out['rev_gap'] = out['rev_score'] - out['rev_best']
    return out


def finish(x, top10, top30, probs, decided):
    x = x.merge(top10.assign(in10=1), how='left', on=IDS)
    x = x[x['in10'].isna()].drop(columns='in10')
    x = x[~x['candidate_id'].isin(set(decided['candidate_id']))]                 # records already matched are skipped
    x = x.merge(top30.rename(columns={'rank': 's1_rank', 'score': 's1_score'}), how='left', on=IDS)
    x['shop_best'] = x['source1_entity_id'].map(probs.groupby('source1_entity_id')['prob'].max()).fillna(0)
    x['n_acc'] = x['source1_entity_id'].map(decided.groupby('source1_entity_id').size()).fillna(0)
    return x.reset_index(drop=True)


def add_ce(x, sources):
    tx = texts_for(np.concatenate([x['source1_entity_id'], x['candidate_id']]), sources)
    a, b = x['source1_entity_id'].map(tx), x['candidate_id'].map(tx)
    ok = (a.notna() & b.notna()).to_numpy()
    x = x[ok].reset_index(drop=True)
    x['ce'] = ce_utils.score(tok, ce, a[ok].tolist(), b[ok].tolist())
    return x


# ---- validation: B1 trains / tunes ----
A, B1, B2 = fold0_split(); B = B1 | B2
ev = pd.read_parquet(f'{WORK}/feats/fold0_B_v7probs.parquet')
base = pd.concat([decide.expected_f05(decide.one_to_one(ev), 0.1)[IDS],
                  pd.read_parquet(f'{WORK}/feats/fold0_B_union1_accepted.parquet')], ignore_index=True)
c30 = pd.read_parquet(f'{WORK}/cands/fold0_top30.parquet', columns=IDS + ['rank', 'score'])
x = finish(rev23(f'{WORK}/cands/reverse_top3_train.parquet', TRAIN[:1], TRAIN[1:], B), ev[IDS], c30[c30['rank'] >= 10], ev, base)
pairs = pd.read_parquet(f'{WORK}/train_pairs.parquet')[['source1_entity_id', 'match_id']].astype(str).rename(columns={'match_id': 'candidate_id'})
x = x.merge(pairs.assign(label=1), how='left', on=IDS)
x['label'] = x['label'].fillna(0).astype('int8')
thr = float(x.loc[x['label'] == 1, 'rev_gap'].quantile(0.01))                 # pre-filter keeping 99% of true copies
x = add_ce(x[x['rev_gap'] >= thr], TRAIN)
b1s = np.array(sorted(B1)); np.random.default_rng(6).shuffle(b1s)
b1_tr = set(b1s[:int(.8 * len(b1s))])
t_mask = x['source1_entity_id'].isin(b1_tr).to_numpy()
v_mask = x['source1_entity_id'].isin(B1).to_numpy() & ~t_mask
params = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=15, min_data_in_leaf=50, feature_fraction=0.9, verbose=-1, seed=8)
addon2 = lgb.train(params, lgb.Dataset(x.loc[t_mask, FEX2], x.loc[t_mask, 'label']), num_boost_round=1000,
                   valid_sets=[lgb.Dataset(x.loc[v_mask, FEX2], x.loc[v_mask, 'label'])], callbacks=[lgb.early_stopping(50)])
addon2.save_model(f'{WORK}/models/lgb_union_addon2.txt')
x['p'] = addon2.predict(x[FEX2], num_iteration=addon2.best_iteration)
gt = pd.read_parquet(f'{WORK}/gt.parquet').astype(str)
truth = evaluate.truth_from_gt(gt[gt['source1_entity_id'].isin(B1)])
def f05_b1(t):
    add = x[(x['p'] >= t) & x['source1_entity_id'].isin(B1)].sort_values('p', ascending=False).drop_duplicates('candidate_id')
    return evaluate.macro_f05(decide.to_pred(pd.concat([base[base['source1_entity_id'].isin(B1)], add[IDS]])), truth)
best_t = max([0.5, 0.6, 0.7, 0.8, 0.9, 0.95], key=f05_b1)
json.dump({'thr': thr, 'best_t': best_t}, open(f'{WORK}/models/union_addon2.json', 'w'))
del ev, base, c30, x, pairs, gt; gc.collect()

# ---- test: v9 decisions (stage 4 + union-1 extras), then reverse #2/#3 extras ----
s4 = pd.concat([pd.read_parquet(p, columns=IDS + ['prob']) for p in sorted(glob.glob(f'{WORK}/test_preds_s4/part_*.parquet'))], ignore_index=True)
sel9 = decide.expected_f05(decide.one_to_one(s4), 0.1)[IDS]
del s4; gc.collect()
u1 = pd.read_parquet(f'{WORK}/test_union_extras.parquet')
v9 = pd.concat([sel9, u1.loc[u1['accept'] & ~u1['candidate_id'].isin(set(sel9['candidate_id'])), IDS]], ignore_index=True)
s7 = pd.concat([pd.read_parquet(p, columns=['source1_entity_id', 'prob']) for p in sorted(glob.glob(f'{WORK}/test_preds_s3v2/part_*.parquet'))])
t30 = pd.read_parquet(f'{WORK}/cands/test_top30.parquet', columns=IDS + ['rank', 'score'])
xt = finish(rev23(f'{WORK}/cands/reverse_top3_test.parquet', TEST[:1], TEST[1:]), t30.loc[t30['rank'] < 10, IDS],
            t30[t30['rank'] >= 10], s7, v9)
del s7, t30; gc.collect()
xt = add_ce(xt[xt['rev_gap'] >= thr], TEST)
xt['p'] = addon2.predict(xt[FEX2], num_iteration=addon2.best_iteration)
acc = xt[xt['p'] >= best_t].sort_values('p', ascending=False).drop_duplicates('candidate_id')
xt['accept'] = False
xt.loc[acc.index, 'accept'] = True
xt[IDS + ['p', 'accept']].to_parquet(f'{WORK}/test_union2_extras.parquet', index=False)
print(f'test extras scored {len(xt):,}, accepted {xt["accept"].sum():,}')
