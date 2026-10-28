"""Business-address normalization (with France-only rules; training data has no France)."""
import re
from normalize import clean_text

_ORDINAL      = re.compile(r'\b(\d+)(st|nd|rd|th)\b')
_DIGIT_LETTER = re.compile(r'(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)')

WORD_NUMBERS = {'first': '1', 'second': '2', 'third': '3', 'fourth': '4', 'fifth': '5',
                'sixth': '6', 'seventh': '7', 'eighth': '8', 'ninth': '9', 'tenth': '10',
                'ground': '0'}
ABBREV = {
    'st': 'street', 'str': 'street', 'rd': 'road', 'ave': 'avenue', 'av': 'avenue',
    'blvd': 'boulevard', 'bd': 'boulevard', 'dr': 'drive', 'ln': 'lane', 'ct': 'court',
    'pl': 'place', 'pkwy': 'parkway', 'hwy': 'highway', 'cir': 'circle', 'ter': 'terrace',
    'trl': 'trail', 'sq': 'square', 'ste': 'suite', 'apt': 'apartment', 'apts': 'apartment',
    'bldg': 'building', 'fl': 'floor', 'flr': 'floor', 'sec': 'sector', 'sect': 'sector',
    'ph': 'phase', 'soc': 'society', 'extn': 'extension', 'ext': 'extension',
    'stn': 'station', 'dist': 'district', 'mkt': 'market', 'rly': 'railway',
    'nr': 'near', 'opp': 'opposite',
}
STOP = {'no', 'number', 'unit', 'suite', 'apartment', 'flat', 'floor', 'plot',
        'house', 'the', 'of', 'and', 'h', 'to'}
LANDMARK = {'near', 'opposite', 'behind', 'beside', 'besides', 'next', 'adjacent',
            'facing', 'landmark', 'pres'}

# France only
FRANCE_RULES = False   # see names.py
ABBREV_FR = {'r': 'rue', 'bld': 'boulevard', 'bvd': 'boulevard', 'all': 'allee', 'imp': 'impasse',
             'ch': 'chemin', 'chem': 'chemin', 'rte': 'route', 'fg': 'faubourg', 'fbg': 'faubourg',
             'qu': 'quai', 'crs': 'cours', 'res': 'residence', 'resid': 'residence'}
STOP_FR = {'de', 'du', 'des', 'la', 'le', 'les', 'd', 'l', 'et', 'cedex', 'bis', 'ter', 'en', 'sur'}
FR_REGIONS = {
    'hauts de france': 'hdf', 'nord pas de calais': 'hdf', 'picardie': 'hdf',
    'nouvelle aquitaine': 'naq', 'aquitaine': 'naq', 'limousin': 'naq', 'poitou charentes': 'naq',
    'ile de france': 'idf', 'auvergne rhone alpes': 'ara', 'rhone alpes': 'ara', 'auvergne': 'ara',
    'occitanie': 'occ', 'midi pyrenees': 'occ', 'languedoc roussillon': 'occ',
    'grand est': 'ges', 'alsace': 'ges', 'lorraine': 'ges', 'champagne ardenne': 'ges',
    'bretagne': 'bre', 'normandie': 'nor', 'basse normandie': 'nor', 'haute normandie': 'nor',
    'pays de la loire': 'pdl', 'provence alpes cote dazur': 'pac', 'provence alpes cote d azur': 'pac', 'paca': 'pac',
    'bourgogne franche comte': 'bfc', 'bourgogne': 'bfc', 'franche comte': 'bfc',
    'centre val de loire': 'cvl', 'centre': 'cvl', 'corse': 'cor',
}

US_STATES = {
    'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar', 'california': 'ca',
    'colorado': 'co', 'connecticut': 'ct', 'delaware': 'de', 'florida': 'fl', 'georgia': 'ga',
    'hawaii': 'hi', 'idaho': 'id', 'illinois': 'il', 'indiana': 'in', 'iowa': 'ia',
    'kansas': 'ks', 'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md',
    'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn', 'mississippi': 'ms',
    'missouri': 'mo', 'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv',
    'new hampshire': 'nh', 'new jersey': 'nj', 'new mexico': 'nm', 'new york': 'ny',
    'north carolina': 'nc', 'north dakota': 'nd', 'ohio': 'oh', 'oklahoma': 'ok',
    'oregon': 'or', 'pennsylvania': 'pa', 'rhode island': 'ri', 'south carolina': 'sc',
    'south dakota': 'sd', 'tennessee': 'tn', 'texas': 'tx', 'utah': 'ut', 'vermont': 'vt',
    'virginia': 'va', 'washington': 'wa', 'west virginia': 'wv', 'wisconsin': 'wi',
    'wyoming': 'wy', 'district of columbia': 'dc', 'washington dc': 'dc', 'puerto rico': 'pr',
}
INDIA_STATES = {
    'andhra pradesh': 'ap', 'arunachal pradesh': 'ar', 'assam': 'as', 'bihar': 'br',
    'chhattisgarh': 'cg', 'goa': 'ga', 'gujarat': 'gj', 'haryana': 'hr',
    'himachal pradesh': 'hp', 'jharkhand': 'jh', 'karnataka': 'ka', 'kerala': 'kl',
    'madhya pradesh': 'mp', 'maharashtra': 'mh', 'manipur': 'mn', 'meghalaya': 'ml',
    'mizoram': 'mz', 'nagaland': 'nl', 'odisha': 'od', 'orissa': 'od', 'punjab': 'pb',
    'rajasthan': 'rj', 'sikkim': 'sk', 'tamil nadu': 'tn', 'tamilnadu': 'tn',
    'telangana': 'tg', 'tripura': 'tr', 'uttar pradesh': 'up', 'uttarakhand': 'uk',
    'uttaranchal': 'uk', 'west bengal': 'wb', 'delhi': 'dl', 'nct of delhi': 'dl',
    'jammu and kashmir': 'jk', 'chandigarh': 'ch', 'puducherry': 'py', 'pondicherry': 'py',
    'ladakh': 'la',
}


def _build(full_to_code, extra_codes=()):
    m = dict(full_to_code)
    for code in set(full_to_code.values()) | set(extra_codes):
        m[code] = code
    return m


STATE_MAPS = {
    'US':     _build(US_STATES),
    'India':  _build(INDIA_STATES, extra_codes=('ts', 'or', 'od', 'mh', 'tn', 'ka')),
    'France': dict(FR_REGIONS),
}
STATE_MAPS['India']['ts'] = 'tg'
STATE_MAPS['India']['or'] = 'od'


def normalize_address(raw: str, country: str = ''):
    """Return (words, numbers, state, landmark) for one address."""
    if not raw:
        return '', '', '', 0
    france = FRANCE_RULES and country == 'France'
    abbrev = {**ABBREV, **ABBREV_FR} if france else ABBREV
    stop = STOP | STOP_FR if france else STOP
    states = {} if (country == 'France' and not FRANCE_RULES) else STATE_MAPS.get(country, {})
    state, kept = '', []
    for part in raw.split(','):
        p = clean_text(part)
        if not p:
            continue
        if p in states:
            state = state or states[p]
            continue
        kept.append(p)
    s = _ORDINAL.sub(r'\1', ' '.join(kept))
    s = _DIGIT_LETTER.sub(' ', s)
    words, numbers, landmark = [], [], 0
    for t in s.split():
        t = WORD_NUMBERS.get(t, t)
        t = abbrev.get(t, t)
        if t in LANDMARK:
            landmark = 1
            continue
        if t in stop:
            continue
        if t.isdigit():
            numbers.append(str(int(t)))
        else:
            words.append(t)
    return ' '.join(words), ' '.join(sorted(set(numbers))), state, landmark


def add_state_aliases(country: str, aliases: dict):
    STATE_MAPS.setdefault(country, {}).update(aliases)
