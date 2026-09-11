"""Download Spark-X2.5-4B from its public ModelScope repository."""
import argparse
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="models/Spark-X2.5-4B")
    args = p.parse_args()
    subprocess.run(["modelscope", "download", "--model", "XHToken/Spark-X2.5-4B", "--local_dir", args.output_dir], check=True)


if __name__ == "__main__":
    main()
