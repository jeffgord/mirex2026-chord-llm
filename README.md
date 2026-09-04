# mirex2026-chord-llm

The chord-llm system estimates the key of a musical passage by running audio through a chord recognizer, and then processing the output with an LLM. For music where the chord representation is uniformative, the system falls back to a deep learning approach.

As such, chord-llm incorporates a few external components:
- We use a chord recognizer from ISMIR 2019 available under MIT license [here](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition) [1]
- We use `Gemini 3.1 Flash Lite` through Google's `genai` API for the LLM call. Note that the audio is never passed through the LLM -- only the extracted chords, as well as chroma activations averaged across the whole passage (see `\method\chord-llm-prompt.txt`).
- As a fallback, we use the AllConv model [2] as implemented in `madmom` ([BSD-licensed](https://github.com/CPJKU/madmom?tab=License-1-ov-file)). The model weights are distributed under [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) for non-commercial use only.

An informal writeup detailing the motivation behind this system along with preliminary results is available here: https://jeffgord.github.io/llm-key-detection/.

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

Then, copy the provided `.env` to the repo root. The submitters will email this file directly to the MIREX2026 organizers. Please do not share this environment file, as it contains a Gemini API key.

## Command line calling format

Run the key detector on a audio file like so:
```
python run.py <input> <output>
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


> **Special Notices:** For the Gemini call, the system requires network access and a valid `GEMINI_API_KEY` (provided via `.env`). Also, CUDA GPU used automatically when available.

## References

[1] J. Jiang, K. Chen, W. Li, and G. Xia, "Large-vocabulary chord transcription via chord structure decomposition," in *Proceedings of the 20th International Society for Music Information Retrieval Conference (ISMIR 2019)*, Delft, The Netherlands, 2019, pp. 644–651. Available: https://archives.ismir.net/ismir2019/paper/000078.pdf

[2] F. Korzeniowski and G. Widmer, "Genre-agnostic key classification with convolutional neural networks," in *Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR 2018)*, Paris, France, 2018. Available: https://ismir2018.ircam.fr/doc/pdfs/7_Paper.pdf