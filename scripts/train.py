"""Launch a LlamaFactory SFT job with paths supplied on the command line."""
import argparse
import subprocess
import tempfile
from pathlib import Path

import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/spark4b_lora.yaml"))
    p.add_argument("--model-path", type=Path)
    p.add_argument("--dataset-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--launcher", default="llamafactory-cli")
    args = p.parse_args()
    config = yaml.safe_load(args.config.read_text())
    for key, value in (("model_name_or_path", args.model_path), ("dataset_dir", args.dataset_dir), ("output_dir", args.output_dir)):
        if value is not None:
            config[key] = str(value)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
        temporary = handle.name
    try:
        subprocess.run([args.launcher, "train", temporary], check=True)
    finally:
        Path(temporary).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
