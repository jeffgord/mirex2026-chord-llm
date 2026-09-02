# mirex2026-chord-llm

## Setup

Requires Python 3.14 (`brew install python@3.14` if you don't have it).

Follow these steps to create a virtual environment and install dependencies:
```
python3.14 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

pip install --no-build-isolation "madmom @ git+https://github.com/CPJKU/madmom.git@27f032e8947204902c675e5e341a3faf5dc86dae"
```
*Note that `madmom` must be installed separately