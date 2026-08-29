# Before the Black Box — Baseline

Minimal baseline for an Agentic AI capstone project on sparse-evidence aviation incident triage.

## What the Baseline Does

The baseline is intentionally simple. It makes **one LLM call** using the complete evidence available in an input case.

Given ACARS-style fault messages and basic flight context, the model is asked to return:

1. ranked incident hypotheses;
2. evidence supporting each hypothesis;
3. important uncertainties;
4. FDR parameters to examine next; and
5. CVR clues to examine next.

The baseline has **no tools, memory, retrieval, or iterative investigation**. It cannot request evidence and then revise its hypotheses. These capabilities will be added in the agentic version of the project.

## Setup

Python 3.10+ and Ollama are required.

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Download the baseline model:

```bash
ollama pull qwen3:4b
```

Verify that the model runs locally:

```bash
ollama run qwen3:4b
```

No API key is required. The baseline model runs locally through Ollama.

## Configuration

The baseline configuration is stored in:

```text
config/baseline.yaml
```

The configuration specifies the local model, temperature, and fixed prompt used for the experiment.


## Running the Baseline

From the repository root:

```bash
python run_baseline.py --input examples/test1.json
```

The program reads the incident, sends it to the model using the fixed baseline prompt, prints the response, and saves the response in:

```text
outputs/test1_output.txt
```

## Test Case

The included test case is:

```text
examples/test1.json
```

It represents a synthetic high-altitude incident containing several ACARS-style warnings and limited contextual information.

## Repository Structure

```text
README.md
requirements.txt
run_baseline.py
config/baseline.yaml
prompts/baseline.txt
examples/test1.json
outputs/
```

## Known Limitations

This is a single-call baseline rather than an investigative agent. It receives all initial evidence at once and cannot interactively request additional FDR/CVR evidence, use external references, or revise its hypotheses after receiving new information.

The final system will use this baseline as a comparison point for evaluating whether iterative agentic investigation improves evidence selection, hypothesis revision, grounding, and uncertainty handling.
