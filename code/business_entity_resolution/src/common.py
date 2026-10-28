"""Shared paths and helpers for the pipeline steps. Paths can be overridden with environment variables."""
import os
import glob
import numpy as np
import pandas as pd
from anyascii import anyascii

DATA = os.environ.get('ER_DATA', '/content/drive/MyDrive/amazon_ml/student_resource/dataset')
WORK = os.environ.get('ER_WORK', '/content/drive/MyDrive/amazon_ml/artifacts')
OUT = os.environ.get('ER_OUT', '/content/drive/MyDrive/amazon_ml/output')
CLEAN = f'{WORK}/clean'
TRAIN, TEST = ['tr_s1', 'tr_s2', 'tr_s3'], ['te_s1', 'te_s2', 'te_s3']
RAW_FILES = {'tr_s1': 'train/train_source1.tsv', 'tr_s2': 'train/train_source2.tsv', 'tr_s3': 'train/train_source3.tsv',
             'te_s1': 'test/test_source1.tsv', 'te_s2': 'test/test_source2.tsv', 'te_s3': 'test/test_source3.tsv'}
N_DOCS = {'train': 2_206_821 + 5_034_616 + 5_285_603, 'test': 1_732_544 + 4_887_273 + 5_082_316}
CE_BAND = (0.01, 0.99)                      # stage-1 probability band scored by the cross-encoder


def ids_of(sources):
    """Entity ids in the same order the cleaned part files are read everywhere in the pipeline."""
    return np.concatenate([pd.read_parquet(p, columns=['entity_id'])['entity_id'].astype(str).to_numpy()
                           for s in sources for p in sorted(glob.glob(f'{CLEAN}/{s}/*.parquet'))])


def texts_for(ids, sources):
    """'name | address' raw text (transliterated to Latin if needed) for the given ids."""
    ids = pd.Index(pd.unique(np.asarray(list(ids), dtype=object)))
    out = {}
    for s in sources:
        r = pd.read_parquet(f'{WORK}/{s}.parquet', columns=['entity_id', 'business_name', 'business_address'],
                            dtype_backend='pyarrow')
        r = r[r['entity_id'].isin(ids)].astype(str)
        for i, nm, ad in zip(r['entity_id'], r['business_name'], r['business_address']):
            t = f'{nm} | {ad}'
            out[i] = t if t.isascii() else anyascii(t)
    return out


def fold0_split():
    """Fold-0 shops -> A (60%), B1 (20%), B2 (20%); the split used for stages 2-4 and the union add-on."""
    s = pd.read_parquet(f'{WORK}/feats/fold0_top10.parquet', columns=['source1_entity_id'])['source1_entity_id']
    shops = np.array(sorted(s.unique()))
    np.random.default_rng(1).shuffle(shops)
    n = len(shops)
    return set(shops[:int(.6 * n)]), set(shops[int(.6 * n):int(.8 * n)]), set(shops[int(.8 * n):])


def add_reverse_features(df, rev, s_ids, p_ids):
    """Reverse-direction features: is this shop the record's own top choice among all S1 shops?"""
    rv = rev[np.isin(rev['q_row'].to_numpy(), pd.Index(p_ids).get_indexer(df['candidate_id'].unique()))].copy()
    rv['candidate_id'] = p_ids[rv['q_row'].to_numpy()]
    rv['source1_entity_id'] = s_ids[rv['p_row'].to_numpy()]
    best = rv[rv['rank'] == 0].set_index('candidate_id')['score']
    second = rv[rv['rank'] == 1].set_index('candidate_id')['score']
    df = df.merge(rv[['source1_entity_id', 'candidate_id', 'rank', 'score']]
                  .rename(columns={'rank': 'rev_rank', 'score': 'rev_score'}),
                  on=['source1_entity_id', 'candidate_id'], how='left')
    df['rev_rank'] = df['rev_rank'].fillna(3).astype('float32')
    df['rev_best'] = df['candidate_id'].map(best).astype('float32')
    df['rev_second'] = df['candidate_id'].map(second).fillna(0).astype('float32')
    df['rev_gap'] = (df['rev_score'] - df['rev_best']).astype('float32')
    df['rev_margin'] = (df['rev_best'] - df['rev_second']).astype('float32')
    return df
