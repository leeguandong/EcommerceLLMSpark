"""Interactive or one-shot generation with Spark-X2.5-4B and an optional LoRA adapter."""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str, adapter_path: str | None):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    return tokenizer, model.eval()


def generate(tokenizer, model, prompt: str, thinking: bool, max_new_tokens: int) -> str:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=thinking,
    )
    device = next(model.parameters()).device
    inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.inference_mode():
        ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                             eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--adapter-path")
    p.add_argument("--prompt")
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    args = p.parse_args()
    tokenizer, model = load_model(args.model_path, args.adapter_path)
    while True:
        prompt = args.prompt if args.prompt is not None else input("问题（exit 退出）：").strip()
        if prompt.lower() in {"exit", "quit"}:
            break
        if prompt:
            print(generate(tokenizer, model, prompt, args.thinking, args.max_new_tokens), flush=True)
        if args.prompt is not None:
            break


if __name__ == "__main__":
    main()
