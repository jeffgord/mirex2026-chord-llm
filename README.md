# mirex2026-chord-llm

The chord-llm system estimates the key of a musical passage by running audio through a chord recognizer, and then processing the output with an LLM. For music where the chord representation is uniformative, the system falls back to a deep learning approach.

As such, chord-llm incorporates a few external components:
- A chord recognizer from ISMIR 2019 available under MIT license [here](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition) [1]
- `Gemini 3.1 Flash Lite` through Google's `genai` API for the LLM call. Note that neither the audio itself nor detailed audio features are passed through the API. The LLM only processes the extracted chords, as well as chroma activations averaged across the whole passage (see `\method\chord-llm-prompt.txt`). This means Google would not be able to recover the original audio from a private evaluation set.
- For the fallback, we use the AllConv model [2] as implemented in `madmom` ([BSD-licensed](https://github.com/CPJKU/madmom?tab=License-1-ov-file)). The model weights are distributed under [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) for non-commercial use only.

An informal writeup detailing the motivation behind this system along with preliminary results is available here: https://jeffgord.github.io/llm-key-detection/.

## Setup

For MIREX2026, the task organizers should first email Jeff (jeffrey.gordon@nyu.edu) for the `.env` file. This contains a Gemini API key, and should not be shared outside the competition. Copy the provided `.env` to the repo root after cloning.

Next, setup the environment. Note that chord-llm is designed for [Python 3.14](https://www.python.org/downloads/release/python-3140/).

After installing Python 3.14, run these commands to create a virtual environment and install dependencies:
```
python3.14 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

pip install --no-build-isolation "madmom @ git+https://github.com/CPJKU/madmom.git@27f032e8947204902c675e5e341a3faf5dc86dae"
```
`madmom` MUST be installed after other dependencies (as shown above).

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
- **Expected memory footprint:** ~1.0 GB peak RSS for 60-second clips
- **Expected runtime:** ~12s per 60-second clip
- **Scratch disk space:** None


> **Special Notices:** For the Gemini call, the system requires network access and a valid `GEMINI_API_KEY` (provided via `.env`). Also, CUDA GPU used automatically when available.

## References

[1] J. Jiang, K. Chen, W. Li, and G. Xia, "Large-vocabulary chord transcription via chord structure decomposition," in *Proceedings of the 20th International Society for Music Information Retrieval Conference (ISMIR 2019)*, Delft, The Netherlands, 2019, pp. 644–651. Available: https://archives.ismir.net/ismir2019/paper/000078.pdf

[2] F. Korzeniowski and G. Widmer, "Genre-agnostic key classification with convolutional neural networks," in *Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR 2018)*, Paris, France, 2018. Available: https://ismir2018.ircam.fr/doc/pdfs/7_Paper.pdf