import argparse
from pathlib import Path

from method.hybrid import predict_key

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIREX Audio Key Detection')
    parser.add_argument('input', type=Path, help='Input WAV file')
    parser.add_argument('output', type=Path, help='Output file for the predicted key')
    args = parser.parse_args()

    key = predict_key(args.input)
    args.output.write_text(key.to_mirex_format())