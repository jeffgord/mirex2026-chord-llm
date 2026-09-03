from dataclasses import dataclass
import re
from typing import Literal
from music21 import key as m21key

TONICS = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

@dataclass
class Key:
    tonic: str
    mode: Literal["major", "minor"]

    def __post_init__(self):
        if self.tonic not in TONICS:
            raise ValueError(f"Invalid tonic: {self.tonic}. Must be one of {TONICS}.")
        if self.mode not in ("major", "minor"):
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'major' or 'minor'.")

    def build(key_str: str) -> Key:
        raw_tonic, raw_mode = key_str.split()
        k = m21key.Key(raw_tonic, raw_mode)
        pc = k.tonic.pitchClass
        tonic = TONICS[pc]
        mode = k.mode.lower()
        return Key(tonic, mode)

    def to_mirex_format(self) -> str:
        return f"{self.tonic}\t{self.mode}\n"

def get_proportion_N(chords: str) -> float:
    entries = re.findall(r"([^\s,][^,]*?)\s+([\d.]+)s", chords)
    total = 0.0
    n_total = 0.0

    for label, dur in entries:
        dur = float(dur)
        total += dur
        if label == "N":
            n_total += dur
    return (n_total / total) if total else None
