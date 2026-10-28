"""Decision layer: turn pair probabilities into final match lists."""
import numpy as np
import pandas as pd


def one_to_one(df):
    """Keep a pair only if this shop is the candidate's highest-probability shop."""
    best = df.groupby('candidate_id')['prob'].transform('max')
    return df[df['prob'] >= best]


def two_thresholds(df, t_first, t_extra):
    """Best candidate needs prob >= t_first; others need prob >= t_extra
    (and are only kept if the shop's best candidate was kept)."""
    df = df.sort_values(['source1_entity_id', 'prob'], ascending=[True, False])
    r = df.groupby('source1_entity_id').cumcount().to_numpy()
    first_ok = df.groupby('source1_entity_id')['prob'].transform('max').to_numpy() >= t_first
    keep = np.where(r == 0, first_ok, first_ok & (df['prob'].to_numpy() >= t_extra))
    return df[keep]


def expected_f05(df, miss=0.0):
    """Per shop, keep the top-k (k may be 0) that maximizes approximate expected F0.5.
    miss = expected true matches outside the candidate list (blocking misses)."""
    df = df.sort_values(['source1_entity_id', 'prob'], ascending=[True, False]).copy()
    g = df.groupby('source1_entity_id')['prob']
    p = df['prob'].clip(0, 1 - 1e-6)
    k = g.cumcount().to_numpy() + 1
    cum_tp = g.cumsum().to_numpy()
    expected_true = g.transform('sum').to_numpy() + miss
    df['Fk'] = 1.25 * cum_tp / (k + 0.25 * expected_true)
    df['k'] = k
    # F0.5 of predicting nothing = probability that the shop has no match at all
    df['F0'] = np.exp(np.log1p(-p).groupby(df['source1_entity_id']).transform('sum').to_numpy() - miss)
    Fmax = df.groupby('source1_entity_id')['Fk'].transform('max')
    kstar = df[df['Fk'] >= Fmax].groupby('source1_entity_id')['k'].min()
    df['kstar'] = df['source1_entity_id'].map(kstar)
    keep = (df['k'] <= df['kstar']) & (Fmax > df['F0'])
    return df[keep.to_numpy()]


def to_pred(sel):
    return sel.groupby('source1_entity_id')['candidate_id'].apply(set).to_dict()
