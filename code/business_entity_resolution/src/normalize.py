"""Text normalization for business entity resolution."""
import re
import ftfy
from anyascii import anyascii

_APOSTROPHE = re.compile(r"['`]")
_NON_ALNUM  = re.compile(r'[^a-z0-9 ]+')
_SPACES     = re.compile(r'\s+')


def clean_text(s: str) -> str:
    """Generic cleaning used for both names and addresses.

    1. Repair broken encoding (ftfy), only when non-ASCII characters exist.
    2. Transliterate any script to Latin and strip accents (anyascii).
    3. Lowercase, turn '&' into 'and', drop apostrophes,
       replace all other punctuation with spaces, collapse whitespace.
    """
    if not s:
        return ''
    if not s.isascii():
        s = anyascii(ftfy.fix_text(s))
    s = s.lower().replace('&', ' and ')
    s = _APOSTROPHE.sub('', s)
    s = _NON_ALNUM.sub(' ', s)
    return _SPACES.sub(' ', s).strip()
