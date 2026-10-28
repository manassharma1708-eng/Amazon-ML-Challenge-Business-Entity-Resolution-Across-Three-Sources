"""Exact competition metric: macro F0.5 over Source 1 entities."""


def parse_ids(s: str) -> set:
    """'S2-1,S3-5' -> {'S2-1', 'S3-5'};  '' -> empty set."""
    return set(s.split(',')) if s else set()


def f05(pred: set, true: set) -> float:
    """F0.5 for one Source 1 entity, including the singleton rules."""
    if not true:                        # singleton
        return 1.0 if not pred else 0.0
    if not pred:                        # real matches exist, we predicted nothing
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(true)
    return 1.25 * precision * recall / (0.25 * precision + recall)


def macro_f05(pred: dict, truth: dict) -> float:
    """pred, truth: {source1_id: set_of_ids}. Every truth entity is scored;
    an entity missing from pred counts as an empty prediction."""
    return sum(f05(pred.get(k, set()), v) for k, v in truth.items()) / len(truth)


def truth_from_gt(gt_df) -> dict:
    return {k: parse_ids(v) for k, v in zip(gt_df['source1_entity_id'], gt_df['matched_entity_ids'])}
