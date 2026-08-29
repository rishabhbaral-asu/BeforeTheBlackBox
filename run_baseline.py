import argparse
import json
from pathlib import Path

import ollama
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    print("Loading configuration...")

    with open("config/baseline.yaml", "r") as f:
        config = yaml.safe_load(f)

    input_path = Path(args.input)

    print(f"Loading incident: {input_path}")

    with open(input_path, "r") as f:
        incident = json.load(f)

    with open(config["prompt_file"], "r") as f:
        prompt_template = f.read()

    prompt = prompt_template.replace(
        "{incident}",
        json.dumps(incident, indent=2)
    )

    print(f"Model: {config['model']}")
    print("Sending incident to local model...")
    print("Waiting for response...\n")

    response = ollama.chat(
        model=config["model"],
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": config["temperature"],
            "num_ctx": config["num_ctx"]
        },
        think = False
    )

    output = response["message"]["content"]

    print("=" * 60)
    print("BASELINE OUTPUT")
    print("=" * 60)
    print(output)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"{input_path.stem}_output.txt"
    output_path.write_text(output)

    print(f"\nSaved output to: {output_path}")


if __name__ == "__main__":
    main()