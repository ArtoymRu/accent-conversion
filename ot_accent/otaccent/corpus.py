"""Corpora: who speaks, with which accent, and which phone each instant should be.

Content is canonical: phones come from the transcript by forced alignment
(shipped with both corpora), never from clustering the frames.
"""
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PHONES = ("AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG "
          "OW OY P R S SH T TH UH UW V W Y Z ZH").split()
PHONE_ID = {p: i for i, p in enumerate(PHONES)}
SILENCE = -1
_ALIASES = {"AX": "AH", "AXR": "ER", "IX": "IH"}

L2_ARCTIC = {
    "arabic": ["ABA", "SKA", "YBAA", "ZHAA"],
    "mandarin": ["BWC", "LXC", "NCC", "TXHC"],
    "hindi": ["ASI", "RRBI", "SVBI", "TNI"],
    "korean": ["HJK", "HKK", "YDCK", "YKWK"],
    "spanish": ["EBVS", "ERMS", "MBMPS", "NJS"],
    "vietnamese": ["HQTV", "PNV", "THV", "TLV"],
}
CMU_ARCTIC = {"native": ["bdl", "slt", "clb", "rms"]}


def phone_id(mark: str) -> int:
    """ARPAbet id of an alignment label; stress digits dropped; anything else is silence."""
    m = re.sub(r"\d", "", mark.strip().upper())
    return PHONE_ID.get(_ALIASES.get(m, m), SILENCE)


@dataclass(frozen=True)
class Utterance:
    speaker: str
    accent: str
    utt: str            # e.g. "arctic_a0001"; the same id is the same sentence for every speaker
    wav: Path
    align: Path

    def text(self) -> str:
        """Reference transcript."""
        if self.align.suffix == ".lab":
            return _festvox_prompts(self.wav.parents[1] / "etc" / "txt.done.data")[self.utt]
        return (self.wav.parents[1] / "transcript" / f"{self.utt}.txt").read_text().strip()

    def phones(self) -> list[tuple[float, float, int]]:
        """[(start_s, end_s, phone_id)], silence included with id -1."""
        if self.align.suffix == ".lab":
            return _festvox_lab(self.align)
        return _textgrid(self.align)


def _textgrid(path: Path):
    import textgrid
    tiers = textgrid.TextGrid.fromFile(str(path)).tiers
    tier = next((t for t in tiers if t.name.lower().startswith("phone")), tiers[-1])
    return [(iv.minTime, iv.maxTime, phone_id(iv.mark)) for iv in tier]


def _festvox_lab(path: Path):
    """CMU ARCTIC labels: a header closed by '#', then lines 'end_time colour phone'."""
    out, start, body = [], 0.0, False
    for line in path.read_text().splitlines():
        if line.strip() == "#":
            body = True
        elif body and line.strip():
            end, _, phone = line.split()[:3]
            out.append((start, float(end), phone_id(phone)))
            start = float(end)
    return out


@lru_cache(maxsize=None)
def _festvox_prompts(path: Path) -> dict:
    """CMU ARCTIC prompts: lines '( arctic_a0001 "Author of the danger trail." )'."""
    return dict(re.findall(r'\(\s*(\S+)\s+"(.*)"\s*\)', path.read_text()))


def utterances(l2_root, cmu_root) -> list[Utterance]:
    """Every utterance of both corpora that has audio and an alignment."""
    out = []
    for accent, speakers in L2_ARCTIC.items():
        for spk in speakers:
            d = Path(l2_root) / spk                       # flat layout, as released
            if not d.exists():
                d = Path(l2_root) / accent / spk          # or sorted into accent folders
            for wav in sorted((d / "wav").glob("*.wav")):
                align = d / "textgrid" / f"{wav.stem}.TextGrid"
                if align.exists():
                    out.append(Utterance(spk, accent, wav.stem, wav, align))
    for accent, speakers in CMU_ARCTIC.items():
        for spk in speakers:
            d = Path(cmu_root) / f"cmu_us_{spk}_arctic"
            for wav in sorted((d / "wav").glob("*.wav")):
                align = d / "lab" / f"{wav.stem}.lab"
                if align.exists():
                    out.append(Utterance(spk, accent, wav.stem, wav, align))
    return out
