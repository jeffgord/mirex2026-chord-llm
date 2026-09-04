import json
import os
from pathlib import Path
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors
from .utils import Key


load_dotenv()

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
config = types.GenerateContentConfig(
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
    prompt = f.read()

def predict_key(chords: str, chroma: str) -> Key:
    contents = prompt.replace("{CHORDS}", chords).replace("{CHROMA}", chroma)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            chat = client.chats.create(model="gemini-3.1-flash-lite", config=config)
            response = chat.send_message(contents)
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