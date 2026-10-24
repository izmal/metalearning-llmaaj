"""eval: label unseen conversations with the judge prompts saved by adapt."""
import argparse
import json
import os
import sys
from pathlib import Path

from data import load_jsonl, render, write_predictions
from llm import generate, parse_json, pmap

JUDGE_THINKING = os.environ.get("JUDGE_THINKING", "high")


def load_prompts(artifacts):
    return json.loads((Path(artifacts) / "judge_prompts.json").read_text())


def judge(row, prompts):
    text = generate(render(row), system=prompts[row["topic"]], thinking=JUDGE_THINKING, max_tokens=8192)
    try:
        violations = parse_json(text).get("violations") or []
    except ValueError:
        violations = []
    # The label follows from the violations alone, so the explanation matches the decision rule exactly.
    return {"item_id": row["item_id"], "label": "bad" if violations else "good", "violations": violations}


def predict(rows, artifacts):
    prompts = load_prompts(artifacts)
    return pmap(lambda row: judge(row, prompts), rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    results = predict(load_jsonl(args.test), args.artifacts)
    write_predictions(args.out, {r["item_id"]: r["label"] for r in results})
    trace = Path(args.out).with_name("violations.jsonl")
    trace.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    print(f"wrote {len(results)} predictions to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
