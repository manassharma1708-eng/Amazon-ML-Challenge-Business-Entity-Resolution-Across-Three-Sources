"""Learn normalization maps from TRAINING data only (no external data)."""
import json
from collections import Counter, defaultdict
from rapidfuzz import fuzz
from normalize import clean_text
import names
import addresses


def _consistent_map(counts, min_count, min_purity):
    """Keep source -> target only if seen often enough and mostly with the same target."""
    out = {}
    for src, targets in counts.items():
        total = sum(targets.values())
        tgt, n = targets.most_common(1)[0]
        if total >= min_count and n / total >= min_purity and tgt != src:
            out[src] = tgt
    return out


def learn_word_map(s1_names, other_names, min_count=10, min_purity=0.6, min_sim=40):
    """Align words of regional-script names (transliterated) with their true S1 name."""
    counts = defaultdict(Counter)
    for n1, n2 in zip(s1_names, other_names):
        if not n2 or n2.isascii():
            continue
        t1, t2 = names.base_tokens(n1), names.base_tokens(n2)
        if len(t1) != len(t2):              # only align names with the same word count
            continue
        for a, b in zip(t2, t1):
            if a != b and fuzz.ratio(a, b) >= min_sim:   # skip clearly misaligned words
                counts[a][b] += 1
    return _consistent_map(counts, min_count, min_purity)


def learn_state_map(s1_addrs, other_addrs, country, min_count=20, min_purity=0.8):
    """Map regional-script address parts to the state code found in the matched S1 address."""
    known = addresses.STATE_MAPS.get(country, {})
    counts = defaultdict(Counter)
    for a1, a2 in zip(s1_addrs, other_addrs):
        if not a2 or a2.isascii():
            continue
        s1_state = addresses.normalize_address(a1, country)[2]
        if not s1_state:
            continue
        for part in a2.split(','):
            if part.isascii():
                continue
            p = clean_text(part)
            if p and p not in known:
                counts[p][s1_state] += 1
    return _consistent_map(counts, min_count, min_purity)


def save_maps(path, word_map, state_maps):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'word_map': word_map, 'state_maps': state_maps}, f, ensure_ascii=False, indent=1)


def load_maps(path):
    with open(path, encoding='utf-8') as f:
        m = json.load(f)
    names.set_word_map(m['word_map'])
    for country, aliases in m['state_maps'].items():
        addresses.add_state_aliases(country, aliases)
    return m
