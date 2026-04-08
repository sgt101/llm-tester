#!/usr/bin/env python3
"""
Evaluate LLM table-reading accuracy against images from generate_tables.py.

The model is asked to read every cell in the table and to compute the sum of
each row and each column.  Three separate scores are reported:

    cell_score      — fraction of cells whose value was read correctly
    row_sum_score   — fraction of row sums computed correctly
    col_sum_score   — fraction of column sums computed correctly
    overall         — mean of the three scores above

Usage:
    uv run evaluate_table.py config.toml

The [evaluation] table_dir in config.toml must point to a directory
produced by generate_tables.py (i.e. it must contain a validation.json with
a "grid" field in each record).
"""

import json
import re
import sys
import tomllib
from pathlib import Path

from llm_clients import get_client


# ---------------------------------------------------------------------------
# Config (mirrors evaluate.py)
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def resolve_config(cfg: dict, config_path: Path) -> tuple[dict, dict, dict]:
    model_cfg = cfg.get("model", {})
    if "provider" not in model_cfg:
        raise ValueError("config [model] section must include 'provider'.")

    eval_cfg = cfg.get("evaluation", {})
    if "table_dir" not in eval_cfg:
        raise ValueError("config [evaluation] section must include 'table_dir'.")

    table_dir = Path(eval_cfg["table_dir"])
    if not table_dir.is_absolute():
        table_dir = config_path.parent / table_dir
    eval_cfg["table_dir"] = table_dir

    output_cfg = cfg.get("output", {})
    return model_cfg, eval_cfg, output_cfg


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict:
    """Extract the outermost JSON object from a model response."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "")

    # Find the first '{' then use the json decoder to consume exactly one object
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response: {text!r}")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text, start)
    return obj


def _to_number(v: object) -> float | None:
    """Parse v as float, covering both integer ('3') and decimal ('3.7') strings."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def normalise_grid(raw: object, rows: int, cols: int) -> list[list[float | None]]:
    """Coerce the model's grid to a 2-D list of numbers (None where unreadable)."""
    grid: list[list[float | None]] = []
    for r in range(rows):
        row_vals: list[float | None] = []
        for c in range(cols):
            try:
                val = _to_number(raw[r][c])
            except (IndexError, TypeError):
                val = None
            row_vals.append(val)
        grid.append(row_vals)
    return grid


def normalise_sums(raw: object, length: int) -> list[float | None]:
    result: list[float | None] = []
    for i in range(length):
        try:
            result.append(_to_number(raw[i]))
        except (IndexError, TypeError):
            result.append(None)
    return result


# ---------------------------------------------------------------------------
# Ground-truth computation
# ---------------------------------------------------------------------------

def ground_truth_from_grid(str_grid: list[list[str]]) -> tuple[list[list[float]], list[float], list[float]]:
    """Return (num_grid, row_sums, col_sums) derived from the string grid."""
    rows = len(str_grid)
    cols = len(str_grid[0]) if rows else 0
    num_grid = [[float(v) for v in row] for row in str_grid]
    row_sums = [round(sum(num_grid[r]), 10) for r in range(rows)]
    col_sums = [round(sum(num_grid[r][c] for r in range(rows)), 10) for c in range(cols)]
    return num_grid, row_sums, col_sums


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_response(
    predicted: dict,
    actual_int_grid: list[list[float]],
    actual_row_sums: list[float],
    actual_col_sums: list[float],
    rows: int,
    cols: int,
) -> dict:
    pred_grid = normalise_grid(predicted.get("grid"), rows, cols)
    pred_row_sums = normalise_sums(predicted.get("row_sums"), rows)
    pred_col_sums = normalise_sums(predicted.get("col_sums"), cols)

    correct_cells = sum(
        1 for r in range(rows) for c in range(cols)
        if pred_grid[r][c] == actual_int_grid[r][c]
    )
    correct_row_sums = sum(
        1 for r in range(rows) if pred_row_sums[r] == actual_row_sums[r]
    )
    correct_col_sums = sum(
        1 for c in range(cols) if pred_col_sums[c] == actual_col_sums[c]
    )

    cell_score    = round(correct_cells    / (rows * cols), 4)
    row_sum_score = round(correct_row_sums / rows,          4)
    col_sum_score = round(correct_col_sums / cols,          4)
    overall       = round((cell_score + row_sum_score + col_sum_score) / 3, 4)

    return {
        "cell_score":         cell_score,
        "row_sum_score":      row_sum_score,
        "col_sum_score":      col_sum_score,
        "overall":            overall,
        "correct_cells":      correct_cells,
        "total_cells":        rows * cols,
        "correct_row_sums":   correct_row_sums,
        "correct_col_sums":   correct_col_sums,
        "predicted_grid":     pred_grid,
        "predicted_row_sums": pred_row_sums,
        "predicted_col_sums": pred_col_sums,
    }


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(base_prompt: str, rows: int, cols: int) -> str:
    """Append structured-output instructions to the base config prompt."""
    example_grid = [[f"r{r}c{c}" for c in range(cols)] for r in range(rows)]
    example = {
        "grid":     example_grid,
        "row_sums": [f"sum of row {r}" for r in range(rows)],
        "col_sums": [f"sum of col {c}" for c in range(cols)],
    }
    suffix = (
        f"\n\nThe table has {rows} rows and {cols} columns."
        f"\n\nFor every cell read the integer shown. Then compute the sum of each"
        f" row and the sum of each column."
        f"\n\nReturn ONLY a valid JSON object in exactly this format"
        f" (replace placeholders with integers):\n"
        f"{json.dumps(example, indent=2)}"
    )
    return base_prompt.rstrip() + suffix


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

DEFAULT_BASE_PROMPT = (
    "Look at this table carefully and read every number shown in each cell."
)


def evaluate(config_path: str | Path) -> None:
    config_path = Path(config_path).resolve()
    cfg = load_config(config_path)
    model_cfg, eval_cfg, output_cfg = resolve_config(cfg, config_path)

    table_dir: Path = eval_cfg["table_dir"]
    base_prompt: str = eval_cfg.get("prompt_table", DEFAULT_BASE_PROMPT)

    # Load and validate the validation file
    validation_path = table_dir / "validation.json"
    if not validation_path.exists():
        print(f"Error: validation.json not found in {table_dir}")
        sys.exit(1)
    ground_truth: list[dict] = json.loads(validation_path.read_text())

    if not ground_truth or "grid" not in ground_truth[0]:
        print("Error: validation.json does not contain table grid data.")
        print("       Run generate_tables.py to produce compatible images.")
        sys.exit(1)

    rows: int = ground_truth[0]["rows"]
    cols: int = ground_truth[0]["cols"]

    client = get_client(
        provider=model_cfg["provider"],
        model=model_cfg.get("name"),
        api_key=model_cfg.get("api_key"),
        temperature=model_cfg.get("temperature", 0.0),
        max_tokens=model_cfg.get("max_tokens", 2048),
    )
    print(f"Provider : {client.provider_name}")
    print(f"Model    : {client.model}")
    print(f"Directory: {table_dir}")
    print(f"Grid     : {rows} rows × {cols} cols  ({rows * cols} cells)")
    print(f"Images   : {len(ground_truth)}\n")

    full_prompt = build_prompt(base_prompt, rows, cols)
    prompt_path = table_dir / "prompt_table.txt"
    prompt_path.write_text(full_prompt, encoding="utf-8")
    print(f"Prompt   : {prompt_path}\n")

    model_slug = re.sub(r"[^a-zA-Z0-9]+", "-", client.model).strip("-")
    log_path  = table_dir / f"log_table-{model_slug}.jsonl"
    log_file  = log_path.open("w", encoding="utf-8")

    results: list[dict] = []
    totals = {"cell_score": 0.0, "row_sum_score": 0.0, "col_sum_score": 0.0, "overall": 0.0}
    evaluated = 0

    for entry in ground_truth:
        fname      = entry["filename"]
        image_path = table_dir / fname

        if not image_path.exists():
            print(f"  SKIP {fname} (file not found)")
            continue

        actual_int_grid, actual_row_sums, actual_col_sums = ground_truth_from_grid(entry["grid"])

        print(f"  [{evaluated + 1}/{len(ground_truth)}] {fname} ...", end=" ", flush=True)

        raw_text    = ""
        parse_error = None
        predicted   = {}
        try:
            response = client.analyze_image(image_path, full_prompt)
            raw_text  = response.text
            predicted = extract_json(raw_text)
        except Exception as exc:
            parse_error = str(exc)
            print(f"ERROR: {exc}")

        log_file.write(json.dumps({
            "filename":         fname,
            "raw_response":     raw_text,
            "predicted":        predicted,
            "actual_grid":      entry["grid"],
            "actual_row_sums":  actual_row_sums,
            "actual_col_sums":  actual_col_sums,
            "error":            parse_error,
        }) + "\n")
        log_file.flush()

        scores = score_response(predicted, actual_int_grid, actual_row_sums, actual_col_sums, rows, cols)

        record: dict = {
            "filename":         fname,
            "actual_grid":      entry["grid"],
            "actual_row_sums":  actual_row_sums,
            "actual_col_sums":  actual_col_sums,
        }
        record.update(scores)
        results.append(record)

        for k in totals:
            totals[k] += scores[k]
        evaluated += 1

        if parse_error is None:
            print(
                f"cells={scores['cell_score']:.2f}  "
                f"row_sums={scores['row_sum_score']:.2f}  "
                f"col_sums={scores['col_sum_score']:.2f}  "
                f"overall={scores['overall']:.2f}"
            )

    # Summary
    if evaluated > 0:
        summary: dict = {"filename": "__summary__", "total_images": evaluated}
        for k in totals:
            summary[k] = round(totals[k] / evaluated, 4)
        results.append(summary)

        print(f"\nSummary:")
        print(f"  {'cell_score':20s} {summary['cell_score']:.4f}")
        print(f"  {'row_sum_score':20s} {summary['row_sum_score']:.4f}")
        print(f"  {'col_sum_score':20s} {summary['col_sum_score']:.4f}")
        print(f"  {'overall':20s} {summary['overall']:.4f}")

    log_file.close()
    print(f"\nLog file : {log_path}")

    score_path_default = table_dir / f"score_table-{model_slug}.json"
    score_path = Path(output_cfg.get("score_file", score_path_default))
    if not score_path.is_absolute():
        score_path = table_dir / score_path
    score_path.write_text(json.dumps(results, indent=2))
    print(f"Score file: {score_path}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: uv run evaluate_table.py <config.toml>")
        sys.exit(1)
    evaluate(sys.argv[1])


if __name__ == "__main__":
    main()
