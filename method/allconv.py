from pathlib import Path
import madmom.features.key as madmom_key
from .utils import Key

proc = madmom_key.CNNKeyRecognitionProcessor()

def predict_key(audio_path: Path) -> Key:
    predictions = proc(str(audio_path))
    key_str = madmom_key.key_prediction_to_label(predictions)
    return Key.build(key_str)
