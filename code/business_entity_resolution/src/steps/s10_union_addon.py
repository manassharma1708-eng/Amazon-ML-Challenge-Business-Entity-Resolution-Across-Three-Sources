"""Step 10 (GPU): union add-on - score each record's reverse #1 shop when that pair is not in the top 10."""
import os, sys, gc, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import WORK, TRAIN, TEST, ids_of, texts_for, fold0_split
import ce_utils, decide

ADD_T = 0.7                                            # middle of the stable plateau found on B1 (0.5-0.8)
FEX = ['ce', 'rev_score', 'rev_second', 'rev_margin', 's1_rank', 's1_score', 'shop_best', 'n_acc']
tok, ce = ce_utils.load_ce(f'{WORK}/models/cross_encoder_v2')


def extras_for(rev_path, s_srcs, p_srcs, shop_filter, top10, top30):
    s_ids, p_ids = ids_of(s_srcs), ids_of(p_srcs)
    rv = pd.read_parquet(rev_path)
    r1 = rv[rv['rank'] == 1].set_index('q_row')['score']
    r0 = rv[rv['rank'] == 0]
    ex = pd.DataFrame({'source1_entity_id': s_ids[r0['p_row'].to_numpy()], 'candidate_id': p_ids[r0['q_row'].to_numpy()],
                       'rev_score': r0['score'].to_numpy(), 'rev_second': r0['q_row'].map(r1).fillna(0).to_numpy()})
    if shop_filter is not None:
        ex = ex[ex['source1_entity_id'].isin(shop_filter)]
    ex = ex.merge(top10.assign(in10=1), how='left', on=['source1_entity_id', 'candidate_id'])
    ex = ex[ex['in10'].isna()].drop(columns='in10')
    ex = ex.merge(top30.rename(columns={'rank': 's1_rank', 'score': 's1_score'}), how='left', on=['source1_entity_id', 'candidate_id'])
    return ex.reset_index(drop=True)


def add_context(ex, probs, sel):
    ex['shop_best'] = ex['source1_entity_id'].map(probs.groupby('source1_entity_id')['prob'].max()).fillna(0)
    ex['n_acc'] = ex['source1_entity_id'].map(sel.groupby('source1_entity_id').size()).fillna(0)
    ex['rev_margin'] = ex['rev_score'] - ex['rev_second']
    return ex[~ex['candidate_id'].isin(set(sel['candidate_id']))].reset_index(drop=True)


# ---- train on B1 extras ----
A, B1, B2 = fold0_split()
ev = pd.read_parquet(f'{WORK}/feats/fold0_B_v7probs.parquet')
c30 = pd.read_parquet(f'{WORK}/cands/fold0_top30.parquet', columns=['source1_entity_id', 'candidate_id', 'rank', 'score'])
ex = extras_for(f'{WORK}/cands/reverse_top3_train.parquet', TRAIN[:1], TRAIN[1:], B1 | B2,
                ev[['source1_entity_id', 'candidate_id']], c30[c30['rank'] >= 10])
pairs = pd.read_parquet(f'{WORK}/train_pairs.parquet')[['source1_entity_id', 'match_id']].astype(str).rename(columns={'match_id': 'candidate_id'})
ex = ex.merge(pairs.assign(label=1), how='left', on=['source1_entity_id', 'candidate_id'])
ex['label'] = ex['label'].fillna(0).astype('int8')
tx = texts_for(np.concatenate([ex['source1_entity_id'], ex['candidate_id']]), TRAIN)
ex['ce'] = ce_utils.score(tok, ce, [tx[i] for i in ex['source1_entity_id']], [tx[i] for i in ex['candidate_id']])
sel = decide.expected_f05(decide.one_to_one(ev), 0.1)
ex = add_context(ex, ev, sel)
b1s = np.array(sorted(B1)); np.random.default_rng(3).shuffle(b1s)
b1_tr = set(b1s[:int(.8 * len(b1s))])
t_mask = ex['source1_entity_id'].isin(b1_tr).to_numpy()
v_mask = ex['source1_entity_id'].isin(B1).to_numpy() & ~t_mask
params = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=15, min_data_in_leaf=50,
              feature_fraction=0.9, verbose=-1, seed=5)
addon = lgb.train(params, lgb.Dataset(ex.loc[t_mask, FEX], ex.loc[t_mask, 'label']), num_boost_round=1000,
                  valid_sets=[lgb.Dataset(ex.loc[v_mask, FEX], ex.loc[v_mask, 'label'])], callbacks=[lgb.early_stopping(50)])
addon.save_model(f'{WORK}/models/lgb_union_addon.txt')
ex['p_add'] = addon.predict(ex[FEX], num_iteration=addon.best_iteration)
ex.loc[ex['p_add'] >= ADD_T, ['source1_entity_id', 'candidate_id']].to_parquet(
    f'{WORK}/feats/fold0_B_union1_accepted.parquet', index=False)
del ev, ex, tx, c30, pairs; gc.collect()

# ---- test ----
s7 = pd.concat([pd.read_parquet(p, columns=['source1_entity_id', 'candidate_id', 'prob'])
                for p in sorted(glob.glob(f'{WORK}/test_preds_s3v2/part_*.parquet'))], ignore_index=True)
sel7 = decide.expected_f05(decide.one_to_one(s7), 0.1)
t30 = pd.read_parquet(f'{WORK}/cands/test_top30.parquet', columns=['source1_entity_id', 'candidate_id', 'rank', 'score'])
ex = extras_for(f'{WORK}/cands/reverse_top3_test.parquet', TEST[:1], TEST[1:], None,
                t30.loc[t30['rank'] < 10, ['source1_entity_id', 'candidate_id']], t30[t30['rank'] >= 10])
del t30; gc.collect()
ex = add_context(ex, s7, sel7)
del s7; gc.collect()
tx = texts_for(np.concatenate([ex['source1_entity_id'], ex['candidate_id']]), TEST)
ex['ce'] = ce_utils.score(tok, ce, [tx[i] for i in ex['source1_entity_id']], [tx[i] for i in ex['candidate_id']])
ex['p_add'] = addon.predict(ex[FEX], num_iteration=addon.best_iteration)
ex['accept'] = ex['p_add'] >= ADD_T
ex[['source1_entity_id', 'candidate_id', 'p_add', 'accept']].to_parquet(f'{WORK}/test_union_extras.parquet', index=False)
print(f'test extras scored {len(ex):,}, accepted {ex["accept"].sum():,}')
