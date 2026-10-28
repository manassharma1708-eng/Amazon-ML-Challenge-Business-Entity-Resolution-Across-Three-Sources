"""End-to-end blocking: cleaned records -> top-K candidate pairs per Source 1 entity."""
import glob
import gc
import numpy as np
import pandas as pd
import blocking_keys as bk
import candidates

COLS = ['entity_id', 'country', 'name_core', 'name_nospace', 'addr_words', 'addr_numbers', 'state']


def word_frequencies(clean_dir, sources):
    """Document frequency of name words and address words over the given sources."""
    name_df = bk.document_frequency([pd.read_parquet(f'{clean_dir}/{s}', columns=['name_core'])['name_core']
                                     for s in sources])
    addr_df = bk.document_frequency([pd.read_parquet(f'{clean_dir}/{s}', columns=['addr_words'])['addr_words']
                                     for s in sources])
    return name_df, addr_df


def build_keys(clean_dir, sources, role, name_df, addr_df, keep_ids=None):
    rows_all, keys_all, ids_all, offset = [], [], [], 0
    for src in sources:
        for part in sorted(glob.glob(f'{clean_dir}/{src}/*.parquet')):
            df = pd.read_parquet(part, columns=COLS)
            if keep_ids is not None:
                df = df[df['entity_id'].isin(keep_ids)]
            r, k = bk.multi_keys(df, name_df, addr_df, role)
            rows_all.append(r + offset); keys_all.append(k)
            ids_all.append(df['entity_id'].astype(str).to_numpy())
            offset += len(df)
            del df; gc.collect()
    return np.concatenate(rows_all), np.concatenate(keys_all), np.concatenate(ids_all)


def run_blocking(clean_dir, s1_source, pool_sources, name_df, addr_df,
                 keep_ids=None, cap=200, top_k=30):
    """Returns (candidates, all query ids, avg unique candidates per query before top-K).
    candidates columns: source1_entity_id, candidate_id, score, n_keys, rank."""
    q_rows, q_keys, q_ids = build_keys(clean_dir, [s1_source], 'query', name_df, addr_df, keep_ids)
    p_rows, p_keys, p_ids = build_keys(clean_dir, pool_sources, 'pool', name_df, addr_df)
    pool = candidates.prepare_pool(p_rows, p_keys, q_keys=q_keys, cap=cap)
    del p_rows, p_keys; gc.collect()
    cands, avg_unique = candidates.generate(q_rows, q_keys, pool, top_k=top_k)
    del pool, q_rows, q_keys; gc.collect()
    cands['source1_entity_id'] = q_ids[cands['q_row'].to_numpy()]
    cands['candidate_id'] = p_ids[cands['p_row'].to_numpy()]
    cands = cands[['source1_entity_id', 'candidate_id', 'score', 'n_keys', 'rank']]
    return cands, q_ids, avg_unique
