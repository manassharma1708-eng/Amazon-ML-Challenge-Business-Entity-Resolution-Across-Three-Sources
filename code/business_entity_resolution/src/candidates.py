"""Candidate generation: join compound keys, score by shared rare keys, keep top-K per query.
Memory-safe: the pool index is built in hash buckets."""
import numpy as np
import pandas as pd


def prepare_pool(p_rows, p_keys, q_keys=None, cap=200, n_buckets=16, chunk=20_000_000):
    """Index pool keys for fast lookup. Keeps only keys with block size <= cap
    (and, if q_keys is given, only keys that some query also has).
    Buckets are defined by the top bits of the hash, so concatenating the sorted
    buckets in order gives one globally sorted index."""
    shift = np.uint64(64 - int(np.log2(n_buckets)))
    bucket = np.empty(len(p_keys), dtype=np.uint8)
    for lo in range(0, len(p_keys), chunk):                  # computed in chunks to save memory
        bucket[lo:lo + chunk] = (p_keys[lo:lo + chunk] >> shift).astype(np.uint8)
    q_uniq = np.unique(q_keys) if q_keys is not None else None

    all_rows, all_uniq, all_start, all_count, offset = [], [], [], [], 0
    for b in range(n_buckets):
        idx = np.flatnonzero(bucket == b)
        k, r = p_keys[idx], p_rows[idx]
        del idx
        order = np.argsort(k, kind='stable')
        k, r = k[order], r[order]
        del order
        uniq, count = np.unique(k, return_counts=True)
        keep = count <= cap
        if q_uniq is not None:
            keep &= np.isin(uniq, q_uniq)
        r = r[np.repeat(keep, count)]                        # rows of the kept keys only
        uniq, count = uniq[keep], count[keep]
        all_rows.append(r); all_uniq.append(uniq); all_count.append(count)
        all_start.append(np.cumsum(count) - count + offset)
        offset += len(r)
        del k, r, uniq, count, keep
    del bucket
    return {'uniq': np.concatenate(all_uniq), 'start': np.concatenate(all_start),
            'count': np.concatenate(all_count), 'rows': np.concatenate(all_rows)}


def generate(q_rows, q_keys, pool, top_k=50, chunk_rows=25_000):
    """Return (candidates DataFrame, average unique candidates per query before top-K).
    Columns: q_row, p_row, score, n_keys, rank (0 = best)."""
    uniq, start, count, rows = pool['uniq'], pool['start'], pool['count'], pool['rows']
    key_weight = 1.0 / np.log2(count + 1.0)                  # block of 1 -> 1.0, block of 200 -> ~0.13
    n_queries = int(q_rows.max()) + 1
    out, total_unique = [], 0

    for lo in range(0, n_queries, chunk_rows):
        m = (q_rows >= lo) & (q_rows < lo + chunk_rows)
        qr, qk = q_rows[m], q_keys[m]
        i = np.minimum(np.searchsorted(uniq, qk), len(uniq) - 1)
        hit = uniq[i] == qk
        qr, i = qr[hit], i[hit]
        lens = count[i]

        q_exp = np.repeat(qr, lens)
        w_exp = np.repeat(key_weight[i], lens).astype(np.float32)
        offsets = np.repeat(start[i] - (np.cumsum(lens) - lens), lens) + np.arange(lens.sum())
        p_exp = rows[offsets]

        df = pd.DataFrame({'q_row': q_exp, 'p_row': p_exp, 'w': w_exp})
        del q_exp, w_exp, offsets, p_exp
        g = df.groupby(['q_row', 'p_row'], sort=False)['w'].agg(['sum', 'size']).reset_index()
        del df
        g.columns = ['q_row', 'p_row', 'score', 'n_keys']
        total_unique += len(g)
        g = g.sort_values(['q_row', 'score'], ascending=[True, False])
        g['rank'] = g.groupby('q_row').cumcount().astype(np.int16)
        out.append(g[g['rank'] < top_k].reset_index(drop=True))
        del g

    return pd.concat(out, ignore_index=True), total_unique / n_queries
