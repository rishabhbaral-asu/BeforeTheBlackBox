import argparse
import json
from pathlib import Path

import ollama
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    # Load configuration
    with open("config/baseline.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load incident
    input_path = Path(args.input)

    with open(input_path, "r") as f:
        incident = json.load(f)

    # Load fixed baseline prompt
    with open(config["prompt_file"], "r") as f:
        prompt_template = f.read()

    prompt = prompt_template.replace(
        "{incident}",
        json.dumps(incident, indent=2)
    )

    # ---------------------------------------------------------
    # MINIMAL BASELINE:
    # One local model call.
    # No tools, RAG, memory, or iterative investigation.
    # ---------------------------------------------------------

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
        think=False
    )

    output = response["message"]["content"]

    # Display output
    print(output)

    # Save output
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"{input_path.stem}_output.txt"
    output_path.write_text(output)

    print(f"\nSaved output to: {output_path}")


if __name__ == "__main__":
    main()