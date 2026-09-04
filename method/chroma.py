from pathlib import Path

import librosa
import numpy as np
import scipy.ndimage


PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def _extract_chroma_vec(audio_path: Path) -> np.ndarray:
    y, sr = librosa.load(str(audio_path))
    y_harm = librosa.effects.harmonic(y, margin=8)
    chroma_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    chroma_filtered = librosa.decompose.nn_filter(
        chroma_harm, aggregate=np.median, metric="cosine"
    )
    chroma_nl_filtered = np.minimum(chroma_harm, chroma_filtered)
    chroma_smooth = scipy.ndimage.median_filter(chroma_nl_filtered, size=(1, 9))
    return np.mean(chroma_smooth, axis=1)

def _format_chroma(vec: np.ndarray) -> str:
    normalized = vec / vec.sum()
    return ", ".join(f"{name}:{val:.3f}" for name, val in zip(PITCH_CLASSES, normalized))

def get_chroma(audio_path: Path) -> str:
    chroma_vec = _extract_chroma_vec(audio_path)
    return _format_chroma(chroma_vec)