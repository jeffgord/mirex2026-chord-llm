from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .allconv import predict_key as allconv_predict_key
from .chord_recognizer.predict import predict_chords
from .chord_llm import predict_key as chord_llm_predict_key
from .chroma import get_chroma
from .utils import Key, get_proportion_N

def predict_key(audio_path: Path) -> Key:
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Run AllConv in the background while we get the Chord-LLM prediction
        allconv_future = executor.submit(allconv_predict_key, audio_path)
        chords_future = executor.submit(predict_chords, audio_path)
        chroma_future = executor.submit(get_chroma, audio_path)

        chords = chords_future.result()
        chroma = chroma_future.result()
        chord_llm_key = chord_llm_predict_key(chords, chroma)
        allconv_key = allconv_future.result()

    # Decision Logic
    if get_proportion_N(chords) >= 0.5:
        return allconv_key
    elif allconv_key.tonic == chord_llm_key.tonic:
        return allconv_key
    else:
        return chord_llm_key