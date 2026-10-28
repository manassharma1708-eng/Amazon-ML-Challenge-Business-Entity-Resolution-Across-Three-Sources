"""Blocking keys: many compound keys per record (long format)."""
import numpy as np
import pandas as pd
from collections import Counter


def document_frequency(series_list):
    """How many records contain each word (a word counts once per record)."""
    df = Counter()
    for s in series_list:
        df.update(t for text in s.astype(str) for t in set(text.split()))
    return df


def _rarest(tokens, freq, k, min_len):
    toks = [t for t in set(tokens) if len(t) >= min_len]
    toks.sort(key=lambda t: (freq.get(t, 0), t))              # rarest first, ties alphabetical
    return toks[:k]


def record_keys(c, core, nospace, words, nums, state, name_df, addr_df, role):
    """All blocking keys of one record, as strings. role: 'query' (S1) or 'pool' (S2/S3)."""
    ntoks = _rarest(core.split(), name_df, 2, 2)
    nparts = list(ntoks) + (['~' + nospace[:5]] if len(nospace) >= 4 else [])
    wtoks = _rarest(words.split(), addr_df, 3, 3)
    numbers = sorted(set(nums.split()), key=lambda x: (-len(x), x))[:2]    # 2 longest numbers
    aparts = wtoks + ['#' + n for n in numbers] + (['@' + state] if state else [])

    # 1. name x address
    keys = [f'N{c}|{n}|{a}' for n in nparts for a in aparts]
    # 2. address only: number x word, and word x word
    keys += [f'A{c}|#{n}|{w}' for n in numbers for w in wtoks]
    keys += [f'A{c}|{min(a, b)}|{max(a, b)}' for i, a in enumerate(wtoks) for b in wtoks[i + 1:]]
    # 3. name only: every query record, but only pool records with no address at all
    if role == 'query' or not (words or nums):
        if len(ntoks) == 2:
            keys.append(f'E{c}|{min(ntoks)}|{max(ntoks)}')
        keys += [f'E{c}|{n}' for n in nparts]
    return keys


def multi_keys(df, name_df, addr_df, role):
    """Return (row_index, key_hash) arrays for a DataFrame of cleaned records."""
    rows, keys = [], []
    cols = zip(df['country'].astype(str), df['name_core'].astype(str), df['name_nospace'].astype(str),
               df['addr_words'].astype(str), df['addr_numbers'].astype(str), df['state'].astype(str))
    for i, (c, core, nospace, words, nums, state) in enumerate(cols):
        ks = record_keys(c, core, nospace, words, nums, state, name_df, addr_df, role)
        rows.extend([i] * len(ks))
        keys.extend(ks)
    h = pd.util.hash_array(np.asarray(keys, dtype=object)).astype(np.uint64)
    return np.asarray(rows, dtype=np.int32), h
