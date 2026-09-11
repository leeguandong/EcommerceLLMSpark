"""Convert instruction/input/output JSON or JSONL files to ShareGPT JSONL."""
import argparse
import json
from pathlib import Path


def read_rows(path: Path):
    raw = path.read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
        return value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        return [json.loads(line) for line in raw.splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    seen, count = set(), 0
    with args.output.open("w", encoding="utf-8") as out:
        for row in read_rows(args.input):
            if not all(isinstance(row.get(k, ""), str) for k in ("instruction", "input", "output")):
                continue
            prompt = "\n".join(x.strip() for x in (row.get("instruction", ""), row.get("input", "")) if x.strip())
            answer = row.get("output", "").strip()
            key = (prompt, answer)
            if not prompt or not answer or key in seen:
                continue
            seen.add(key)
            out.write(json.dumps({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]}, ensure_ascii=False) + "\n")
            count += 1
    print(json.dumps({"written": count, "output": str(args.output)}))


if __name__ == "__main__":
    main()
