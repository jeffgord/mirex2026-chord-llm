# mirex2026-chord-llm

TODO: Add some brief information detailing the system here. specifically the LLM call

## Environment Setup

Requires Python 3.14 (`brew install python@3.14` if you don't have it).

First, follow these steps to create a virtual environment and install dependencies:
```
python3.14 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

pip install --no-build-isolation "madmom @ git+https://github.com/CPJKU/madmom.git@27f032e8947204902c675e5e341a3faf5dc86dae"
```
Note that `madmom` MUST be installed after other dependencies (as shown above).

Second, copy the provided `.env` to the repo root. The submitters will provide the MIREX2026 organizers with this file over email. This contains a Gemini API key, and so cannot be included in this GitHub. Please do not share this environment file.

## Command line calling format

Run the key detector on a audio file like so
```
python run.py <input.wav> <output>
```

For example:
```
python run.py track01.wav track01.key
```

## Run details

- **Threads/cores:** 3 (fixed, not configurable)
- **Expected memory footprint:** TBD
- **Expected runtime:** TBD
- **Scratch disk space:** None

> **Special notices:** 
- For the Gemini call, the system requires network access and a valid `GEMINI_API_KEY` (provided via `.env`).
- CUDA GPU used automatically when available.