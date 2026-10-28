"""Stage 2: consistency between a shop's candidates, on top of stage-1 probabilities.
A shop's true copies in S2/S3 resemble each other, so a weak candidate that looks like
a confident candidate of the same shop is probably also a match."""
import glob
import gc
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

B_COLS = ['entity_id', 'name_core', 'addr_words', 'addr_numbers']


def load_fields(clean_dir, sources, ids):
    ids = pd.Index(pd.unique(np.asarray(ids, dtype=object)))
    parts = []
    for src in sources:
        for part in sorted(glob.glob(f'{clean_dir}/{src}/*.parquet')):
            d = pd.read_parquet(part, columns=B_COLS)
            parts.append(d[d['entity_id'].isin(ids)].astype(str))
    return pd.concat(parts, ignore_index=True).set_index('entity_id')


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def stage2_features(df, fields, n_anchors=3):
    """df: source1_entity_id, candidate_id, prob (all candidates of a shop together)."""
    d = df[['source1_entity_id', 'candidate_id', 'prob']].reset_index(drop=True)
    shop = d['source1_entity_id']
    g = d.groupby('source1_entity_id')['prob']
    F = pd.DataFrame(index=d.index)

    # --- shop context of the stage-1 probability ---
    pr = g.rank(ascending=False, method='first')
    F['prob'] = d['prob'].astype('float32')
    F['prob_rank'] = pr.astype('float32')
    F['prob_max'] = g.transform('max').astype('float32')
    F['prob_gap'] = (d['prob'] - F['prob_max']).astype('float32')
    psum = g.transform('sum')
    F['prob_sum'] = psum.astype('float32')
    F['prob_share'] = (d['prob'] / psum.clip(lower=1e-6)).astype('float32')
    second = d.loc[pr == 2].set_index('source1_entity_id')['prob']
    F['prob_second'] = shop.map(second).fillna(0).astype('float32')
    F['n_cands'] = g.transform('size').astype('float32')
    F['n_above_05'] = (d['prob'] >= 0.5).groupby(shop).transform('sum').astype('float32')

    # --- support: similarity to the shop's most confident candidates (anchors) ---
    anc = d.loc[pr <= n_anchors, ['source1_entity_id', 'candidate_id', 'prob']].copy()
    anc['anc_rank'] = pr[pr <= n_anchors].to_numpy()
    anc.columns = ['source1_entity_id', 'anc_id', 'anc_prob', 'anc_rank']
    m = d[['source1_entity_id', 'candidate_id']].reset_index().merge(anc, on='source1_entity_id')
    m = m[m['candidate_id'] != m['anc_id']].reset_index(drop=True)

    get = lambda ids, c: fields[c].reindex(ids).fillna('').tolist()
    ci, ai = m['candidate_id'], m['anc_id']
    m['name_sim'] = _cp(get(ci, 'name_core'), get(ai, 'name_core'), fuzz.token_set_ratio)
    m['addr_sim'] = _cp(get(ci, 'addr_words'), get(ai, 'addr_words'), fuzz.token_set_ratio)
    m['num_sim'] = _cp(get(ci, 'addr_numbers'), get(ai, 'addr_numbers'), fuzz.token_set_ratio)
    m['w'] = m['anc_prob'] * (m['name_sim'] + m['addr_sim']) / 200
    m['conf_similar'] = ((m['anc_prob'] >= 0.5) & (m['name_sim'] >= 80) & (m['addr_sim'] >= 80)).astype('float32')
    m['sim_prob'] = np.where((m['name_sim'] >= 80) & (m['addr_sim'] >= 70), m['anc_prob'], 0).astype('float32')

    agg = m.groupby('index').agg(sup_max=('w', 'max'), sup_mean=('w', 'mean'),
                                 sup_n_conf=('conf_similar', 'sum'), sup_simprob=('sim_prob', 'max'))
    first = m.loc[m.groupby('index')['anc_rank'].idxmin(), ['index', 'name_sim', 'addr_sim', 'num_sim']].set_index('index')
    first.columns = ['first_name_sim', 'first_addr_sim', 'first_num_sim']
    F = F.join(agg).join(first)
    return d[['source1_entity_id', 'candidate_id']], F


def run_chunked(preds, clean_dir, pool_sources, n_chunks=4):
    """Stage-2 features for all pairs, in chunks of shops."""
    shops = preds['source1_entity_id'].unique()
    outs = []
    for i, chunk in enumerate(np.array_split(shops, n_chunks)):
        p = preds[preds['source1_entity_id'].isin(set(chunk))]
        fields = load_fields(clean_dir, pool_sources, p['candidate_id'])
        ids, F = stage2_features(p, fields)
        outs.append(pd.concat([ids, F], axis=1))
        del fields, p; gc.collect()
        print(f'  stage-2 chunk {i+1}/{n_chunks}: {len(ids):,} pairs')
    return pd.concat(outs, ignore_index=True)


# ---------------- v2: extra pair features from the error analysis ----------------
import re
from rapidfuzz.distance import Levenshtein

_LEET = str.maketrans('013456789', 'oleasgtbg')
_LEET_RE = re.compile(r'(?<=[a-z])\d|\d(?=[a-z])')


def deleet(s):
    """Digits touching letters -> look-alike letters: 8lanks -> blanks, 6arden -> garden."""
    return _LEET_RE.sub(lambda m: m.group(0).translate(_LEET), s)


def name_counts(clean_dir, sources):
    """How many records share each (country, name_core) key."""
    keys = []
    for src in sources:
        for part in sorted(glob.glob(f'{clean_dir}/{src}/*.parquet')):
            d = pd.read_parquet(part, columns=['country', 'name_core'])
            keys.append(d['country'].astype(str) + '|' + d['name_core'].astype(str))
    return pd.concat(keys, ignore_index=True).value_counts()


def load_s1_fields(clean_dir, source, ids):
    ids = pd.Index(pd.unique(np.asarray(ids, dtype=object)))
    parts = []
    for part in sorted(glob.glob(f'{clean_dir}/{source}/*.parquet')):
        d = pd.read_parquet(part, columns=['entity_id', 'country', 'name_core', 'addr_numbers'])
        parts.append(d[d['entity_id'].isin(ids)].astype(str))
    return pd.concat(parts, ignore_index=True).set_index('entity_id')


def extra_features(ids, a_fields, b_fields, s1_counts, pool_counts):
    a = a_fields.reindex(ids['source1_entity_id'])
    b = b_fields.reindex(ids['candidate_id'])
    a_core, b_core = a['name_core'].fillna('').tolist(), b['name_core'].fillna('').tolist()
    key = pd.Series((a['country'].fillna('') + '|' + a['name_core'].fillna('')).to_numpy())
    E = pd.DataFrame(index=ids.index)
    E['s1_name_dup'] = key.map(s1_counts).fillna(0).to_numpy().astype('float32')
    E['pool_name_dup'] = key.map(pool_counts).fillna(0).to_numpy().astype('float32')
    E['name_exact'] = (np.array(a_core, dtype=object) == np.array(b_core, dtype=object)).astype('float32')
    a_dl, b_dl = [deleet(x) for x in a_core], [deleet(x) for x in b_core]
    E['n_tsr_deleet'] = _cp(a_dl, b_dl, fuzz.token_set_ratio)
    E['ns_ratio_deleet'] = _cp([x.replace(' ', '') for x in a_dl], [x.replace(' ', '') for x in b_dl], fuzz.ratio)
    bw, bn = b['addr_words'].fillna('').tolist(), b['addr_numbers'].fillna('').tolist()
    an = a['addr_numbers'].fillna('').tolist()
    E['b_addr_empty'] = np.array([float(not w and not n) for w, n in zip(bw, bn)], dtype='float32')
    near = []
    for x, y in zip(an, bn):
        X, Y = set(x.split()), set(y.split())
        if not X or not Y:
            near.append(np.nan)
        else:
            near.append(float(any(u != v and min(len(u), len(v)) >= 2 and Levenshtein.distance(u, v) <= 1
                                  for u in X for v in Y)))
    E['num_near'] = np.array(near, dtype='float32')
    return E


def run_chunked_v2(preds, clean_dir, s1_source, pool_sources, s1_counts, pool_counts, n_chunks=4):
    shops = preds['source1_entity_id'].unique()
    outs = []
    for i, chunk in enumerate(np.array_split(shops, n_chunks)):
        p = preds[preds['source1_entity_id'].isin(set(chunk))]
        b_fields = load_fields(clean_dir, pool_sources, p['candidate_id'])
        a_fields = load_s1_fields(clean_dir, s1_source, p['source1_entity_id'])
        ids, F = stage2_features(p, b_fields)
        E = extra_features(ids, a_fields, b_fields, s1_counts, pool_counts)
        outs.append(pd.concat([ids, F, E], axis=1))
        del p, b_fields, a_fields, F, E; gc.collect()
        print(f'  stage-2 v2 chunk {i+1}/{n_chunks}: {len(ids):,} pairs')
    return pd.concat(outs, ignore_index=True)
