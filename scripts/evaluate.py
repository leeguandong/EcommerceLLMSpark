"""Generate resumable predictions for a JSONL file with id and prompt fields."""
import argparse
import json
from pathlib import Path

from chat import generate, load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--adapter-path")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    args = p.parse_args()
    tokenizer, model = load_model(args.model_path, args.adapter_path)
    done = set()
    if args.output.exists():
        done = {json.loads(line)["id"] for line in args.output.read_text().splitlines() if line.strip()}
    with args.output.open("a", encoding="utf-8") as out:
        for line in args.input.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["id"] in done:
                continue
            prediction = generate(tokenizer, model, row["prompt"], False, args.max_new_tokens)
            out.write(json.dumps({"id": row["id"], "prediction": prediction}, ensure_ascii=False) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
