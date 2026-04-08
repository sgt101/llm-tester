#!/usr/bin/env python3
"""
Evaluate LLM counting accuracy against composite images.

Usage:
    uv run evaluate.py config.toml

The TOML config controls which model to use, which composite directory to
evaluate, and the prompt sent to the model.  Results are written to
score.json inside the composite directory.
"""

import json
import re
import sys
import tomllib
from pathlib import Path

from llm_clients import get_client


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def resolve_config(cfg: dict, config_path: Path) -> tuple[dict, dict, dict]:
    """Return (model_cfg, eval_cfg, output_cfg) with defaults filled in."""
    model_cfg = cfg.get("model", {})
    if "provider" not in model_cfg:
        raise ValueError("config [model] section must include 'provider'.")

    eval_cfg = cfg.get("evaluation", {})
    if "composite_dir" not in eval_cfg:
        raise ValueError("config [evaluation] section must include 'composite_dir'.")
    if "prompt" not in eval_cfg:
        raise ValueError("config [evaluation] section must include 'prompt'.")

    # Resolve composite_dir relative to the config file location
    composite_dir = Path(eval_cfg["composite_dir"])
    if not composite_dir.is_absolute():
        composite_dir = config_path.parent / composite_dir
    eval_cfg["composite_dir"] = composite_dir

    output_cfg = cfg.get("output", {})
    return model_cfg, eval_cfg, output_cfg


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict:
    """Extract the first JSON object from a model response string."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "")

    # Find the first {...} block
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text!r}")
    return json.loads(match.group())


def normalise_counts(raw: dict) -> dict[str, int]:
    """Return a dict of {item_name: int_count} from the model's raw JSON."""
    counts: dict[str, int] = {}
    for k, v in raw.items():
        key = str(k).strip().lower()
        try:
            counts[key] = int(v)
        except (TypeError, ValueError):
            counts[key] = 0
    return counts


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_object(predicted: int, actual: int) -> float:
    """
    Partial-credit score for a single object type.

    Score = 1 - abs(predicted - actual) / actual, clamped to [0, 1].

    Examples:
        20 predicted, 23 actual  ->  1 - 3/23  = 0.87
        23 predicted, 23 actual  ->  1.0
         0 predicted, 23 actual  ->  0.0
        30 predicted, 23 actual  ->  1 - 7/23  = 0.70
    When actual is 0: 1.0 if the model also predicted 0, else 0.0.
    """
    if actual == 0:
        return 1.0 if predicted == 0 else 0.0
    return max(0.0, round(1.0 - abs(predicted - actual) / actual, 4))


def score_image(predicted: dict[str, int], actual: dict[str, int]) -> dict[str, float]:
    """Return per-object partial-credit scores. Keys come from the ground-truth dict."""
    return {
        name: score_object(predicted.get(name.lower(), 0), count)
        for name, count in actual.items()
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(config_path: str | Path) -> None:
    config_path = Path(config_path).resolve()
    cfg = load_config(config_path)
    model_cfg, eval_cfg, output_cfg = resolve_config(cfg, config_path)

    composite_dir: Path = eval_cfg["composite_dir"]
    prompt: str = eval_cfg["prompt"]
    target_name: str | None = eval_cfg.get("target")

    # Load validation ground truth
    validation_path = composite_dir / "validation.json"
    if not validation_path.exists():
        print(f"Error: validation.json not found in {composite_dir}")
        sys.exit(1)
    ground_truth: list[dict] = json.loads(validation_path.read_text())
    gt_by_file = {entry["filename"]: entry for entry in ground_truth}

    # Build LLM client
    client = get_client(
        provider=model_cfg["provider"],
        model=model_cfg.get("name"),
        api_key=model_cfg.get("api_key"),
        temperature=model_cfg.get("temperature", 0.0),
        max_tokens=model_cfg.get("max_tokens", 2048),
    )
    print(f"Provider : {client.provider_name}")
    print(f"Model    : {client.model}")
    print(f"Directory: {composite_dir}")
    print(f"Images   : {len(ground_truth)}\n")

    # Determine all object names from the first validation record (excluding 'filename')
    object_names: list[str] = [k for k in ground_truth[0].keys() if k != "filename"]

    # Build the full prompt by appending an auto-generated items section to the
    # base prompt from config.toml.  This ensures the model is told the exact
    # object names that appear in the images rather than relying on hardcoded
    # examples in the config file.
    example_json = json.dumps({name: "N" for name in object_names})
    items_suffix = (
        f"\n\nThe objects to count are: {', '.join(object_names)}.\n"
        f"Return JSON in exactly this format (replace N with the integer count):\n"
        f"{example_json}"
    )
    full_prompt = eval_cfg["prompt"].rstrip() + items_suffix

    # Save the full prompt to a file in the composite directory for auditing.
    prompt_path = composite_dir / "prompt.txt"
    prompt_path.write_text(full_prompt, encoding="utf-8")
    print(f"Prompt   : {prompt_path}")

    results: list[dict] = []
    # Accumulate totals for summary
    totals: dict[str, float] = {name: 0.0 for name in object_names}
    evaluated = 0

    model_slug = re.sub(r"[^a-zA-Z0-9]+", "-", client.model).strip("-")
    log_path = composite_dir / f"log-{model_slug}.jsonl"
    log_file = log_path.open("w", encoding="utf-8")

    for entry in ground_truth:
        fname = entry["filename"]
        image_path = composite_dir / fname

        if not image_path.exists():
            print(f"  SKIP {fname} (file not found)")
            continue

        actual_counts = {name: int(entry[name]) for name in object_names}

        print(f"  [{evaluated + 1}/{len(ground_truth)}] {fname} ...", end=" ", flush=True)
        raw_text = ""
        parse_error = None
        parsed_json = None
        try:
            response = client.analyze_image(image_path, full_prompt)
            raw_text = response.text
            parsed_json = extract_json(raw_text)
            predicted_counts = normalise_counts(parsed_json)
        except Exception as exc:
            parse_error = str(exc)
            print(f"ERROR: {exc}")
            predicted_counts = {}

        log_file.write(json.dumps({
            "filename": fname,
            "raw_response": raw_text,
            "parsed": parsed_json,
            "predicted": predicted_counts,
            "actual": actual_counts,
            "error": parse_error,
        }) + "\n")
        log_file.flush()

        per_object = score_image(predicted_counts, actual_counts)
        object_scores = list(per_object.values())
        overall = sum(object_scores) / len(object_scores) if object_scores else 0.0

        record: dict = {"filename": fname}
        record["predicted"] = {n: predicted_counts.get(n.lower(), None) for n in object_names}
        record["actual"] = actual_counts
        record.update(per_object)
        if target_name and target_name in per_object:
            record["target"] = per_object[target_name]
        record["overall"] = round(overall, 4)
        results.append(record)

        for name in object_names:
            totals[name] += per_object.get(name, 0.0)
        evaluated += 1

        correct_str = ", ".join(
            f"{n}={s:.2f}" for n, s in per_object.items()
        )
        print(f"overall={overall:.2f}  ({correct_str})")

    # Build summary record
    if evaluated > 0:
        summary: dict = {"filename": "__summary__", "total_images": evaluated}
        summary_object_scores = []
        for name in object_names:
            avg = round(totals[name] / evaluated, 4)
            summary[name] = avg
            summary_object_scores.append(avg)
        if target_name and target_name in object_names:
            summary["target"] = summary[target_name]
        summary["overall"] = round(sum(summary_object_scores) / len(summary_object_scores), 4)
        results.append(summary)

        print(f"\nSummary:")
        for name in object_names:
            print(f"  {name:20s} {summary[name]:.4f}")
        if target_name:
            print(f"  {'target':20s} {summary.get('target', 'n/a')}")
        print(f"  {'overall':20s} {summary['overall']:.4f}")

    log_file.close()
    print(f"Log file : {log_path}")

    # Write score.json
    score_path_default = composite_dir / f"score-{model_slug}.json"
    score_path = Path(output_cfg.get("score_file", score_path_default))
    if not score_path.is_absolute():
        score_path = composite_dir / score_path
    score_path.write_text(json.dumps(results, indent=2))
    print(f"\nScore file: {score_path}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: uv run evaluate.py <config.toml>")
        sys.exit(1)
    evaluate(sys.argv[1])


if __name__ == "__main__":
    main()
