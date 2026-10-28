"""Business-name normalization (with France-only rules; training data has no France)."""
import re
from normalize import clean_text

_DOMAIN    = re.compile(r'(?i)\bwww\.|\.(?:co\.in|com|net|org|in|biz|info)\b')
_LEET_ZERO = re.compile(r'(?<=[a-z])0|0(?=[a-z])')
_MESSRS    = re.compile(r'^m s ')

LEGAL = {
    'inc': 'inc', 'incorporated': 'inc', 'incorp': 'inc',
    'corp': 'corp', 'corporation': 'corp',
    'co': 'co', 'company': 'co',
    'llc': 'llc', 'pllc': 'llc',
    'llp': 'llp', 'elelpi': 'llp',
    'lp': 'lp', 'plc': 'plc',
    'ltd': 'ltd', 'limited': 'ltd', 'limitet': 'ltd', 'limitd': 'ltd', 'limirrd': 'ltd',
    'pvt': 'pvt', 'private': 'pvt', 'praivet': 'pvt', 'praibhet': 'pvt',
    'sarl': 'sarl', 'sas': 'sas', 'sasu': 'sas', 'eurl': 'eurl', 'sa': 'sa', 'sci': 'sci', 'snc': 'snc',
}
FILLER = {'mr', 'mrs', 'ms', 'dr', 'smt', 'shri', 'sri', 'shree', 'the', 'services', 'service'}

# France only
FRANCE_RULES = False   # v8 was built WITHOUT the France-only rules (no leaderboard gain)
FR_LEGAL = {'cie': 'co', 'compagnie': 'co', 'scp': 'scp', 'selarl': 'selarl', 'sarlu': 'sarl'}
FR_LEGAL_RUNS = {'eurl', 'sarl', 'sarlu', 'sas', 'sasu', 'sa', 'sci', 'snc', 'scp', 'selarl'}
FR_FILLER = {'le', 'la', 'les', 'l', 'de', 'du', 'des', 'd', 'et', 'ste', 'societe', 'ets', 'etablissements'}

WORD_MAP = {}


def set_word_map(mapping: dict):
    WORD_MAP.clear()
    WORD_MAP.update(mapping)


def base_tokens(raw: str):
    s = _DOMAIN.sub(' ', raw or '')
    s = clean_text(s)
    s = _LEET_ZERO.sub('o', s)
    s = _MESSRS.sub('', s)
    return s.split()


def _join_letter_runs(tokens):
    """'e u r l' -> 'eurl' (only when the letters spell a known legal form)."""
    out, run = [], []
    for t in tokens + [None]:
        if t is not None and len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            joined = ''.join(run)
            out.extend([joined] if joined in FR_LEGAL_RUNS else run)
            run = []
        if t is not None:
            out.append(t)
    return out


def normalize_name(raw: str, country: str = ''):
    """Return (full, core, nospace, legal) versions of a business name."""
    tokens = [WORD_MAP.get(t, t) for t in base_tokens(raw)]
    france = FRANCE_RULES and country == 'France'
    if france:
        tokens = _join_letter_runs(tokens)
    legal_map = {**LEGAL, **FR_LEGAL} if france else LEGAL
    filler = FILLER | FR_FILLER if france else FILLER
    legal = ' '.join(sorted({legal_map[t] for t in tokens if t in legal_map}))
    full = ' '.join(legal_map.get(t, t) for t in tokens)
    core_tokens = [t for t in tokens if t not in legal_map and t not in filler]
    if not core_tokens:
        core_tokens = [t for t in tokens if t not in legal_map] or tokens
    core = ' '.join(core_tokens)
    return full, core, core.replace(' ', ''), legal
