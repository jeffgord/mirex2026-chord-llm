from pathlib import Path

from allconv import predict_key as allconv_predict_key
from chord_llm import predict_key as chord_llm_predict_key
from utils import Key, get_proportion_N

def predict_key(audio_path: Path) -> Key:
    # Run the audio through AllConv
    allconv_key =  allconv_predict_key(audio_path)

    # Also, get Chord-LLM prediction
    chords = ChordRecognizer(audio_path)
    chord_llm_key = chord_llm_predict_key(chords)

    # Decision Logic
    if get_proportion_N(chords) >= 0.5:
        return allconv_key
    elif allconv_key.tonic == chord_llm_key.tonic:
        return allconv_key
    else:
        return chord_llm_key