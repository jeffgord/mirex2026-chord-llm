#!/usr/bin/env bash
# Runs run.py on every audio file under fma-keys, recording per-file
# runtime and peak memory (via /usr/bin/time -l) for the README's
# "Expected memory footprint" / "Expected runtime" fields.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="/Users/jeffreygordon/Dev/llm-key-detection/fma-keys"
OUT_DIR="$SCRIPT_DIR/fma-keys-results"
KEYS_DIR="$OUT_DIR/keys"
CSV="$OUT_DIR/benchmark.csv"
LOG="$OUT_DIR/errors.log"

mkdir -p "$KEYS_DIR"

source "$SCRIPT_DIR/.venv/bin/activate"

if [[ ! -f "$CSV" ]]; then
    echo "file,runtime_sec,peak_mem_bytes,status" > "$CSV"
    : > "$LOG"
fi

# Per-file timeout, since a network call (Gemini) can hang indefinitely
# across a laptop sleep/wake with no error. macOS has no `timeout`/`gtimeout`
# built in, so this is a plain watchdog: kill python (the child of the
# /usr/bin/time wrapper), then the wrapper itself, if it runs too long.
TIMEOUT_SEC=300

FILE_LIST="$(mktemp)"
find "$INPUT_DIR" -type f -iname '*.mp3' | sort > "$FILE_LIST"
TOTAL=$(wc -l < "$FILE_LIST" | tr -d ' ')
echo "Found $TOTAL audio files"

i=0
while IFS= read -r f; do
    i=$((i + 1))
    base="$(basename "${f%.*}")"
    out_file="$KEYS_DIR/$base.key"

    if [[ -f "$out_file" ]]; then
        echo "[$i/$TOTAL] skip $base (already done)"
        continue
    fi

    echo "[$i/$TOTAL] $base"
    timing_file="$(mktemp)"
    /usr/bin/time -l python "$SCRIPT_DIR/run.py" "$f" "$out_file" 2> "$timing_file" &
    pid=$!
    (
        sleep "$TIMEOUT_SEC"
        pkill -9 -P "$pid" 2>/dev/null
        kill -9 "$pid" 2>/dev/null
    ) &
    watcher=$!
    if wait "$pid" 2>/dev/null; then
        status="ok"
    else
        status="error"
        cat "$timing_file" >> "$LOG"
        echo "--- $base failed or exceeded ${TIMEOUT_SEC}s, see above ---" >> "$LOG"
    fi
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null

    real=$(grep -E '^\s*[0-9.]+ real' "$timing_file" | awk '{print $1}')
    mem=$(grep 'peak memory footprint' "$timing_file" | awk '{print $1}')
    echo "$base,${real:-},${mem:-},$status" >> "$CSV"
    rm -f "$timing_file"
done < "$FILE_LIST"
rm -f "$FILE_LIST"

echo "Done. Results in $OUT_DIR"
