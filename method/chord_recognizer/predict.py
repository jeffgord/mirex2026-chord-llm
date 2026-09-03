import contextlib
import sys
from pathlib import Path

import numpy as np

_ISMIR_DIR = Path(__file__).parent
sys.path.insert(0, str(_ISMIR_DIR))

# chordnet_ismir_naive (and NetworkInterface's checkpoint loading below) read
# files using paths relative to the current working directory, not __file__.
with contextlib.chdir(_ISMIR_DIR):
    from chordnet_ismir_naive import ChordNet
    from mir.nn.train import NetworkInterface
from extractors.cqt import CQTV2
from mir import io, DataEntry
from extractors.xhmm_ismir import XHMMDecoder
from settings import DEFAULT_SR, DEFAULT_HOP_LENGTH

_HMM_TEMPLATE = str(_ISMIR_DIR / 'data' / 'submission_chord_list.txt')
MODEL_NAMES = ['joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s%d.best' % i for i in range(5)]

_nets = None
_hmm = None


def _get_nets_and_hmm():
    global _nets, _hmm
    if _nets is None:
        with contextlib.chdir(_ISMIR_DIR):
            _nets = [NetworkInterface(ChordNet(None), name, load_checkpoint=False) for name in MODEL_NAMES]
        _hmm = XHMMDecoder(template_file=_HMM_TEMPLATE)
    return _nets, _hmm


def predict_chords(audio_path: Path) -> str:
    nets, hmm = _get_nets_and_hmm()

    entry = DataEntry()
    entry.prop.set('sr', DEFAULT_SR)
    entry.prop.set('hop_length', DEFAULT_HOP_LENGTH)
    entry.append_file(str(audio_path.resolve()), io.MusicIO, 'music')
    entry.append_extractor(CQTV2, 'cqt')

    probs = [net.inference(entry.cqt) for net in nets]
    probs = [np.mean([p[i] for p in probs], axis=0) for i in range(len(probs[0]))]

    chordlab = hmm.decode_to_chordlab(entry, probs, False)

    return ', '.join(f'{label} {end - start:.3f}s' for start, end, label in chordlab)
