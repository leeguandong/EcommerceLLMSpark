"""Validate model loading and chat-template rendering before a long run."""
import argparse
import json

from chat import load_model, generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--adapter-path")
    args = p.parse_args()
    tokenizer, model = load_model(args.model_path, args.adapter_path)
    answer = generate(tokenizer, model, "请用一句话介绍你自己。", False, 64)
    print(json.dumps({"status": "ok", "tokenizer_vocab": len(tokenizer), "sample": answer}, ensure_ascii=False))


if __name__ == "__main__":
    main()
