"""Pair features for the matching model (language-independent, no country feature)."""
import glob
import math
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

CLEAN_COLS = ['entity_id', 'name_full', 'name_core', 'name_nospace', 'legal',
              'addr_words', 'addr_numbers', 'state', 'landmark']


def load_clean_subset(clean_dir, sources, ids):
    """Cleaned records for the given ids only (read part by part to save memory)."""
    ids = pd.Index(pd.unique(np.asarray(ids, dtype=object)))
    parts = []
    for src in sources:
        for part in sorted(glob.glob(f'{clean_dir}/{src}/*.parquet')):
            d = pd.read_parquet(part, columns=CLEAN_COLS)
            d = d[d['entity_id'].isin(ids)]
            parts.append(d.astype({c: str for c in CLEAN_COLS if c != 'landmark'}))
    return pd.concat(parts, ignore_index=True)


def attach(pairs, left, right):
    """Add a_* (Source 1) and b_* (candidate) cleaned fields to each pair."""
    L = left.add_prefix('a_').rename(columns={'a_entity_id': 'source1_entity_id'})
    R = right.add_prefix('b_').rename(columns={'b_entity_id': 'candidate_id'})
    return pairs.merge(L, on='source1_entity_id', how='left').merge(R, on='candidate_id', how='left')


def _cp(a, b, scorer):
    """Element-wise similarity of two equal-length string lists (C++, multi-threaded)."""
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def compute_features(df, name_df, addr_df, n_docs):
    col = lambda c: df[c].astype(str).tolist()
    a_core, b_core = col('a_name_core'), col('b_name_core')
    a_ns, b_ns = col('a_name_nospace'), col('b_name_nospace')
    a_full, b_full = col('a_name_full'), col('b_name_full')
    a_w, b_w = col('a_addr_words'), col('b_addr_words')
    a_n, b_n = col('a_addr_numbers'), col('b_addr_numbers')
    a_l, b_l = col('a_legal'), col('b_legal')
    a_st, b_st = col('a_state'), col('b_state')

    F = pd.DataFrame(index=df.index)
    shop = df['source1_entity_id']

    # --- 1. blocking signals ---
    F['score'] = df['score'].astype('float32')
    F['n_keys'] = df['n_keys'].astype('int16')
    F['rank'] = df['rank'].astype('int16')
    g = df.groupby('source1_entity_id')['score']
    F['rel_score'] = (df['score'] / g.transform('max')).astype('float32')
    F['score_share'] = (df['score'] / g.transform('sum')).astype('float32')
    F['n_cands'] = g.transform('size').astype('int16')
    F['from_s3'] = df['candidate_id'].str.startswith('S3').astype('int8')

    # --- 2. name string similarities ---
    F['n_tsr'] = _cp(a_core, b_core, fuzz.token_set_ratio)
    F['n_tsort'] = _cp(a_core, b_core, fuzz.token_sort_ratio)
    F['n_ratio'] = _cp(a_core, b_core, fuzz.ratio)
    F['n_full_tsr'] = _cp(a_full, b_full, fuzz.token_set_ratio)
    F['ns_partial'] = _cp(a_ns, b_ns, fuzz.partial_ratio)
    F['ns_jw'] = _cp(a_ns, b_ns, JaroWinkler.normalized_similarity)

    # --- 3. address string similarities (NaN if either address has no words) ---
    a_tsr = _cp(a_w, b_w, fuzz.token_set_ratio)
    a_part = _cp(a_w, b_w, fuzz.partial_ratio)
    no_words = np.array([(not x) or (not y) for x, y in zip(a_w, b_w)])
    a_tsr[no_words] = np.nan
    a_part[no_words] = np.nan
    F['a_tsr'], F['a_partial'] = a_tsr, a_part

    # --- 4. token-level features (Python loop) ---
    cache_n, cache_a = {}, {}
    def idf(t, freq, cache):
        v = cache.get(t)
        if v is None:
            v = math.log(n_docs / (1 + freq.get(t, 0)))
            cache[t] = v
        return v

    def idf_jacc(x, y, freq, cache):
        tx, ty = set(x.split()), set(y.split())
        if not tx or not ty:
            return np.nan
        inter = sum(idf(t, freq, cache) for t in tx & ty)
        union = sum(idf(t, freq, cache) for t in tx | ty)
        return inter / union if union else np.nan

    out = {k: [] for k in ['n_idf', 'first_eq', 'acronym', 'n_tok_a', 'n_tok_b', 'legal_both',
                           'legal_conflict', 'a_idf', 'num_shared', 'num_jacc', 'main_num_eq',
                           'a_has_num', 'b_has_num', 'state_eq', 'b_addr_empty']}
    for c1, c2, ns2, w1, w2, n1, n2, l1, l2, s1, s2 in zip(a_core, b_core, b_ns, a_w, b_w, a_n, b_n, a_l, b_l, a_st, b_st):
        t1, t2 = c1.split(), c2.split()
        out['n_idf'].append(idf_jacc(c1, c2, name_df, cache_n))
        out['first_eq'].append(float(bool(t1) and bool(t2) and t1[0] == t2[0]))
        initials = ''.join(t[0] for t in t1)
        out['acronym'].append(float(len(t1) >= 2 and initials == ns2))
        out['n_tok_a'].append(len(t1)); out['n_tok_b'].append(len(t2))
        L1, L2 = set(l1.split()), set(l2.split())
        out['legal_both'].append(float(bool(L1) and bool(L2)))
        out['legal_conflict'].append(float(bool(L1) and bool(L2) and not (L1 & L2)))
        out['a_idf'].append(idf_jacc(w1, w2, addr_df, cache_a))
        N1, N2 = set(n1.split()), set(n2.split())
        out['a_has_num'].append(float(bool(N1))); out['b_has_num'].append(float(bool(N2)))
        if N1 and N2:
            out['num_shared'].append(len(N1 & N2))
            out['num_jacc'].append(len(N1 & N2) / len(N1 | N2))
            out['main_num_eq'].append(float(max(N1, key=len) == max(N2, key=len)))
        else:
            out['num_shared'].append(np.nan); out['num_jacc'].append(np.nan); out['main_num_eq'].append(np.nan)
        out['state_eq'].append(float(s1 == s2) if (s1 and s2) else np.nan)
        out['b_addr_empty'].append(float(not w2 and not n2))
    for k, v in out.items():
        F[k] = np.asarray(v, dtype=np.float32)

    F['a_landmark'] = df['a_landmark'].astype('int8')
    F['b_landmark'] = df['b_landmark'].astype('int8')

    # --- 5. context within the shop: how does this candidate compare to the others? ---
    for f in ['n_tsr', 'n_idf', 'a_tsr', 'ns_jw']:
        grp = F[f].groupby(shop)
        F[f'{f}_rank'] = grp.rank(ascending=False, method='min').astype('float32')
        F[f'{f}_gap'] = (F[f] - grp.transform('max')).astype('float32')
    return F
