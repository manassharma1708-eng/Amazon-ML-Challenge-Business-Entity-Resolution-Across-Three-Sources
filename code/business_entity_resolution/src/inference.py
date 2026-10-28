"""Test-time inference: features + model probabilities in shop chunks, then output files."""
import os
import gc
import csv
import numpy as np
import pandas as pd
import features


def score_candidates(cands, clean_dir, s1_source, pool_sources, name_df, addr_df, n_docs,
                     model, feat_names, n_chunks=8, out_dir=None):
    """Features + probabilities for every candidate pair, processed in chunks of shops.
    A shop's candidates always stay in the same chunk (the in-shop context features need that).
    Finished chunks are saved in out_dir and skipped on re-run (resumable)."""
    shops = cands['source1_entity_id'].unique()
    outs = []
    for i, chunk in enumerate(np.array_split(shops, n_chunks)):
        path = f'{out_dir}/part_{i:02d}.parquet' if out_dir else None
        if path and os.path.exists(path):
            outs.append(pd.read_parquet(path))
            print(f'chunk {i+1}/{n_chunks}: already done')
            continue
        p = cands[cands['source1_entity_id'].isin(set(chunk))].reset_index(drop=True)
        left = features.load_clean_subset(clean_dir, [s1_source], p['source1_entity_id'])
        right = features.load_clean_subset(clean_dir, pool_sources, p['candidate_id'])
        df = features.attach(p, left, right)
        del left, right; gc.collect()
        F = features.compute_features(df, name_df, addr_df, n_docs)
        del df; gc.collect()
        prob = model.predict(F[feat_names])
        out = p[['source1_entity_id', 'candidate_id']].assign(prob=prob.astype('float32'))
        if path:
            out.to_parquet(path, index=False)
        outs.append(out)
        print(f'chunk {i+1}/{n_chunks}: {len(out):,} pairs scored')
        del p, F; gc.collect()
    return pd.concat(outs, ignore_index=True)


def write_lists(all_ids, pairs, list_col, path):
    """One row per Source 1 id; comma-separated ids (order kept, duplicates removed), '' if none."""
    lists = pairs.groupby('source1_entity_id', sort=False)['candidate_id'].apply(lambda s: ','.join(dict.fromkeys(s)))
    out = pd.DataFrame({'source1_entity_id': all_ids})
    out[list_col] = out['source1_entity_id'].map(lists).fillna('')
    out.to_csv(path, sep='\t', index=False, quoting=csv.QUOTE_NONE)
    return out
