"""Load model-input JSONL and render one example as text for prompts."""
import csv
import json
from pathlib import Path

TOPICS = ("credit", "financial_position", "savings", "spending")


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def render(row):
    lines = ["<context>", json.dumps(row["context"], ensure_ascii=False, indent=1), "</context>", "<conversation>"]
    for message in row["conversation"]:
        lines.append(f"[{message['role']}]: {message['content']}")
    lines.append("</conversation>")
    return "\n".join(lines)


def write_predictions(path, labels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["item_id", "label"])
        for item_id, label in labels.items():
            writer.writerow([item_id, label])
