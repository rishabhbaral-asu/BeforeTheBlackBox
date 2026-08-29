import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------

    print("Loading configuration...")

    with open("config/baseline.yaml", "r") as f:
        config = yaml.safe_load(f)

    input_path = Path(args.input)

    # ---------------------------------------------------------
    # Load incident
    # ---------------------------------------------------------

    print(f"Loading incident: {input_path}")

    with open(input_path, "r") as f:
        incident = json.load(f)

    # ---------------------------------------------------------
    # Load prompt
    # ---------------------------------------------------------

    with open(config["prompt_file"], "r") as f:
        prompt_template = f.read()

    prompt = prompt_template.replace(
        "{incident}",
        json.dumps(incident, indent=2)
    )

    # ---------------------------------------------------------
    # Load Hugging Face model
    # ---------------------------------------------------------

    model_name = config["model"]

    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    print("Model loaded.")
    print("Running baseline...\n")

    # ---------------------------------------------------------
    # Format as a chat
    # ---------------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    # ---------------------------------------------------------
    # SINGLE MODEL CALL
    # ---------------------------------------------------------

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=config["max_new_tokens"],
            do_sample=False
        )

    # Remove the input tokens so we only decode the answer
    generated_tokens = generated[
        :, inputs["input_ids"].shape[1]:
    ]

    output = tokenizer.decode(
        generated_tokens[0],
        skip_special_tokens=True
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    print("=" * 60)
    print("BASELINE OUTPUT")
    print("=" * 60)
    print(output)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_path = (
        output_dir /
        f"{input_path.stem}_output.txt"
    )

    output_path.write_text(output)

    print("\n" + "=" * 60)
    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    main()