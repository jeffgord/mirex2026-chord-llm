import json
import os
from pathlib import Path
import time
import librosa
import numpy as np
import scipy.ndimage
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors
from .utils import Key

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def extract_chroma(audio_path: Path) -> np.ndarray:
    y, sr = librosa.load(str(audio_path))
    y_harm = librosa.effects.harmonic(y, margin=8)
    chroma_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    chroma_filtered = librosa.decompose.nn_filter(
        chroma_harm, aggregate=np.median, metric="cosine"
    )
    chroma_nl_filtered = np.minimum(chroma_harm, chroma_filtered)
    chroma_smooth = scipy.ndimage.median_filter(chroma_nl_filtered, size=(1, 9))
    return np.mean(chroma_smooth, axis=1)

def format_chroma(vec: np.ndarray) -> str:
    normalized = vec / vec.sum()
    return ", ".join(f"{name}:{val:.3f}" for name, val in zip(PITCH_CLASSES, normalized))

class GeminiKeyEstimator:
    def __init__(self):
        load_dotenv()
        self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
        self.config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="high"),
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "key": types.Schema(type=types.Type.STRING),
                    "explanation": types.Schema(type=types.Type.STRING),
                },
                required=["explanation", "key"],
            ),
        )

        prompt_path = Path(__file__).parent / 'chord-llm-prompt.txt'
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt = f.read()

    def predict(self, chords: str, chroma: str) -> str:
        contents = self.prompt.replace("{CHORDS}", chords).replace("{CHROMA}", chroma)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=contents,
                    config=self.config,
                )
                response_json = json.loads(response.text)
                return Key.build(response_json["key"])
            except errors.APIError as e:
                if e.code == 503 and attempt < max_retries - 1:
                    time.sleep(30)
                    continue
                raise
            except (json.JSONDecodeError, KeyError, ValueError):
                if attempt < max_retries - 1:
                    continue
                raise

key_estimator = GeminiKeyEstimator()

def predict_key(chords: str, audio_path: Path) -> Key:
    chroma_vec = extract_chroma(audio_path)
    chroma = format_chroma(chroma_vec)

    return key_estimator.predict(chords, chroma)