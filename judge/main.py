"""Container entrypoint: `adapt` (/input/train.jsonl -> /artifacts) or `eval` (/input/test.jsonl + /artifacts -> /output)."""
import os
import sys
from pathlib import Path

INPUT = Path(os.environ.get("INPUT_DIR", "/input"))
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
OUTPUT = Path(os.environ.get("OUTPUT_DIR", "/output"))


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "adapt":
        sys.argv = ["adapt", "--train", str(INPUT / "train.jsonl"), "--artifacts", str(ARTIFACTS)]
        import adapt
        adapt.main()
    elif stage == "eval":
        sys.argv = ["eval", "--test", str(INPUT / "test.jsonl"), "--artifacts", str(ARTIFACTS),
                    "--out", str(OUTPUT / "predictions.csv")]
        import predict
        predict.main()
    else:
        raise SystemExit("usage: main.py adapt|eval")


if __name__ == "__main__":
    main()
