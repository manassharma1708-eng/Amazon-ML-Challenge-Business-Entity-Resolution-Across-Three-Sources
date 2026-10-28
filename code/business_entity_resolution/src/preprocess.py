"""Clean every record once: raw name + address -> normalized fields.

Memory-safe: reads a source file in chunks and writes one cleaned part file per chunk.
Create the worker Pool AFTER learn_maps.load_maps(...) and BEFORE loading any big data.
"""
import os
import pandas as pd
import pyarrow.parquet as pq
import names
import addresses

COLS = ['name_full', 'name_core', 'name_nospace', 'legal',
        'addr_words', 'addr_numbers', 'state', 'landmark']
RAW_COLS = ['entity_id', 'business_name', 'business_address', 'country']


def _clean_row(args):
    name, addr, country = args
    return names.normalize_name(name, country) + addresses.normalize_address(addr, country)


def preprocess(df: pd.DataFrame, pool=None, chunksize: int = 5000) -> pd.DataFrame:
    rows = list(zip(df['business_name'].astype(str),
                    df['business_address'].astype(str),
                    df['country'].astype(str)))
    if pool is None:
        out = [_clean_row(r) for r in rows]
    else:
        out = pool.map(_clean_row, rows, chunksize=chunksize)

    res = pd.DataFrame(out, columns=COLS)
    res.insert(0, 'entity_id', df['entity_id'].astype(str).values)
    res.insert(1, 'country', df['country'].astype(str).values)
    res['landmark'] = res['landmark'].astype('int8')
    text_cols = [c for c in res.columns if c != 'landmark']
    res[text_cols] = res[text_cols].astype('string[pyarrow]')
    return res


def preprocess_file(src_path: str, out_dir: str, pool=None, part_size: int = 500_000):
    """Clean one source file chunk by chunk; skips parts that already exist (resumable)."""
    os.makedirs(out_dir, exist_ok=True)
    pf = pq.ParquetFile(src_path)
    for i, batch in enumerate(pf.iter_batches(batch_size=part_size, columns=RAW_COLS)):
        part_path = os.path.join(out_dir, f'part_{i:03d}.parquet')
        if os.path.exists(part_path):
            print(f'  part {i:03d}: already done')
            continue
        cleaned = preprocess(batch.to_pandas(), pool)
        cleaned.to_parquet(part_path, index=False)
        print(f'  part {i:03d}: {len(cleaned):,} rows saved')
        del cleaned
