"""Step 6: reverse blocking - every S2/S3 record looks up its top-3 S1 shops (all shops compete)."""
import os, sys, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import WORK, CLEAN, TRAIN, TEST
import blocking, candidates

for tag, s1, pool_srcs, srcs in [('train', 'tr_s1', ['tr_s2', 'tr_s3'], TRAIN), ('test', 'te_s1', ['te_s2', 'te_s3'], TEST)]:
    path = f'{WORK}/cands/reverse_top3_{tag}.parquet'
    if os.path.exists(path):
        print(tag, 'already done'); continue
    name_df, addr_df = blocking.word_frequencies(CLEAN, srcs)
    s_rows, s_keys, _ = blocking.build_keys(CLEAN, [s1], 'query', name_df, addr_df)
    index = candidates.prepare_pool(s_rows, s_keys, q_keys=None, cap=200)
    del s_rows, s_keys; gc.collect()
    p_rows, p_keys, _ = blocking.build_keys(CLEAN, pool_srcs, 'pool', name_df, addr_df)
    del name_df, addr_df; gc.collect()
    rev, _ = candidates.generate(p_rows, p_keys, index, top_k=3, chunk_rows=100_000)
    rev[['q_row', 'p_row', 'score', 'rank']].to_parquet(path, index=False)
    print(tag, f'{len(rev):,} rows'); del rev, p_rows, p_keys, index; gc.collect()
