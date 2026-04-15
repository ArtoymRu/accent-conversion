"""
Mapping from L2-ARCTIC speaker codes to accent labels (L1 language).
Based on L2-ARCTIC README speaker table.
"""

# Speaker code -> accent (folder name / label)
SPEAKER_TO_ACCENT = {
    "ABA": "arabic",
    "SKA": "arabic",
    "YBAA": "arabic",
    "ZHAA": "arabic",
    "BWC": "mandarin",
    "LXC": "mandarin",
    "NCC": "mandarin",
    "TXHC": "mandarin",
    "ASI": "hindi",
    "RRBI": "hindi",
    "SVBI": "hindi",
    "TNI": "hindi",
    "HJK": "korean",
    "HKK": "korean",
    "YDCK": "korean",
    "YKWK": "korean",
    "EBVS": "spanish",
    "ERMS": "spanish",
    "MBMPS": "spanish",
    "NJS": "spanish",
    "HQTV": "vietnamese",
    "PNV": "vietnamese",
    "THV": "vietnamese",
    "TLV": "vietnamese",
}

# Ordered list of accents for consistent indexing
ACCENTS = ["arabic", "mandarin", "hindi", "korean", "spanish", "vietnamese"]

# Accent -> list of speaker codes
ACCENT_TO_SPEAKERS = {}
for speaker, accent in SPEAKER_TO_ACCENT.items():
    ACCENT_TO_SPEAKERS.setdefault(accent, []).append(speaker)


def get_accent(speaker_code: str) -> str:
    """Return accent label for a speaker code (e.g. ABA -> arabic)."""
    return SPEAKER_TO_ACCENT.get(speaker_code.upper(), "")


def get_accent_id(accent: str) -> int:
    """Return numeric id for accent (for classification)."""
    try:
        return ACCENTS.index(accent.lower())
    except ValueError:
        return -1
