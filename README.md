# llm-tester

A pipeline for testing how well LLMs can count objects in images and read numbers from tables. This code was generated using Claude Code, and it's provided under an MIT license (see LICENSE)
![The intended flow of using this code](llm_tester_flow.png "What this testing program does")

The toolkit has two independent workflows:

**Object-counting workflow**
1. **`generate_composites.py`** — creates composite images containing pseudo-randomly distributed instances of your source objects, plus a `validation.json` ground-truth file.
2. **`evaluate.py`** — submits those images to an LLM and scores the responses against the ground truth.

**Table-reading workflow**
1. **`generate_tables.py`** — creates grid/table images with randomised numbers in each cell, plus a `validation.json` ground-truth file recording every cell value.
2. **`evaluate_table.py`** — submits those images to an LLM, asks it to read every cell and compute row/column sums, and scores the responses.

**Visualisation**
- **`plot_scores.py`** — renders a bar chart from any `score-*.json` or `score_table-*.json` files for visual comparison across models.

---

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

```bash
uv sync
```

To use local MLX models on Apple Silicon, install the optional MLX dependencies:

```bash
uv sync --extra mlx
```

Place your source images (PNG, JPEG, or GIF) in a folder called `sourceimg/` for the object-counting workflow:

```
llm-tester/
├── generate_composites.py
├── evaluate.py
├── generate_tables.py
├── evaluate_table.py
├── plot_scores.py
├── llm_clients.py
├── config.toml
├── sourceimg/
│   ├── elephant.png
│   ├── giraffe.jpg
│   └── circle.png
└── output/               # created automatically
```

---

### API keys

Set the relevant environment variable for the provider you intend to use:

| Provider | Environment variable |
|----------|----------------------|
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Google | `GEMINI_API_KEY` |
| MLX (local) | — no key required — |

**Option 1 — export in your shell session:**

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="AIza..."
```

You only need to set the variable for the provider you're using. Add the line to your `~/.zshrc` or `~/.bashrc` to make it permanent.

**Option 2 — `.env` file (recommended for local development):**

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
```

Then load it before running:

```bash
set -a && source .env && set +a
uv run evaluate.py config.toml
```

Or use `uv run --env-file .env evaluate.py config.toml` if your `uv` version supports `--env-file`.

> **Note:** Never commit your `.env` file. Add it to `.gitignore`:
> ```bash
> echo ".env" >> .gitignore
> ```

**Option 3 — inline in `config.toml` (not recommended):**

You can also set `api_key` directly in the `[model]` section of `config.toml`, but avoid committing that file with a real key in it.

```toml
[model]
provider = "anthropic"
api_key  = "sk-ant-..."   # omit this line and use an env var instead
```

---

## Object-counting workflow

### Stage 1 — Generate composite images

```bash
uv run generate_composites.py --noobjects <N> --format <fmt> --numbout <N> [options]
```

#### Required flags

| Flag | Description |
|------|-------------|
| `--noobjects N` | Total number of objects placed in each composite image |
| `--format fmt` | Output format: `png`, `jpg`, `jpeg`, `gif`, or `pdf` |
| `--numbout N` | How many composite images to generate |

#### Optional flags

| Flag | Default | Description |
|------|---------|-------------|
| `--layout` | `grid` | Placement style: `grid`, `random`, or `spiral` |
| `--target name` | — | Fix one image type as the "target" (stem name, e.g. `elephant`). Must be used with `--numtarget`. |
| `--numtarget N` | — | Number of target instances fixed in every composite. Must be used with `--target`. |
| `--sourcedir path` | `sourceimg` | Directory containing source images |
| `--canvas WxH` | `1200x900` | Canvas size in pixels |
| `--seed N` | — | Integer seed for reproducible output |

#### Layout modes

| Mode | Behaviour |
|------|-----------|
| `grid` | Objects arranged in a regular grid with slight random jitter |
| `random` | Objects scattered at random positions; retries placement to minimise overlap |
| `spiral` | Objects follow an Archimedean spiral outward from the canvas centre |

#### Output

Images and a ground-truth file are written to `output/<format>_<numbout>_<noobjects>_<layout>/`. The current `config.toml` is also copied into the directory for reproducibility.

```
output/png_10_4_grid/
├── composite_0001.png
├── composite_0002.png
├── ...
├── config.toml
└── validation.json
```

`validation.json` is a JSON array with one record per image:

```json
[
  {"filename": "composite_0001.png", "circle": 2, "elephant": 1, "giraffe": 1},
  {"filename": "composite_0002.png", "circle": 0, "elephant": 3, "giraffe": 1}
]
```

#### Examples

```bash
# 10 PNG composites, 4 objects each, default grid layout
uv run generate_composites.py --noobjects 4 --format png --numbout 10

# Spiral layout, larger canvas
uv run generate_composites.py --noobjects 8 --format jpg --numbout 20 \
    --layout spiral --canvas 1600x1200

# Fix exactly 8 elephants per composite, fill remaining 12 randomly
uv run generate_composites.py --noobjects 20 --format png --numbout 50 \
    --target elephant --numtarget 8 --seed 42
```

---

### Stage 2 — Evaluate with an LLM

#### Configuration (`config.toml`)

```toml
[model]
# Provider: anthropic, openai, google, or mlx
provider = "google"
# Model name — omit to use the provider default
name = "gemini-2.0-flash"
# api_key — omit to read from the environment variable

[evaluation]
# Directory produced by generate_composites.py (must contain validation.json)
composite_dir = "output/png_10_4"

# Optional: marks one object type as the "target" in score output
# target = "elephant"

# Base prompt sent to the model with each image.
# Object names and example JSON format are appended automatically.
prompt = """
Look at this image carefully and count every distinct object type you can see.

Return ONLY a valid JSON object — no explanation, no markdown — where each key
is the object name (lowercase) and each value is the integer count of that
object in the image.
"""

[output]
# Defaults to score-<model-name>.json inside composite_dir
# score_file = "score.json"
```

#### Provider defaults

| Provider | Default model |
|----------|---------------|
| Anthropic | `claude-opus-4-6` |
| OpenAI | `gpt-4o` |
| Google | `gemini-2.0-flash` |
| MLX | `mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit` |

#### Local inference with MLX (Apple Silicon)

The `mlx` provider runs a vision-language model entirely on-device — no API key or internet connection required during inference. Models are downloaded from Hugging Face on first use and cached locally.

```bash
uv sync --extra mlx
```

```toml
[model]
provider = "mlx"
name = "mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit"
```

Browse available models at [huggingface.co/mlx-community](https://huggingface.co/mlx-community). The MLX provider is only supported on Apple Silicon Macs.

#### Run

```bash
uv run evaluate.py config.toml
```

#### Output

Results are written to `<composite_dir>/score-<model-name>.json`:

```json
[
  {
    "filename": "composite_0001.png",
    "predicted": {"circle": 3, "elephant": 2},
    "actual":    {"circle": 5, "elephant": 2},
    "circle": 0.6,
    "elephant": 1.0,
    "overall": 0.8
  },
  {
    "filename": "__summary__",
    "total_images": 10,
    "circle": 0.75,
    "elephant": 0.8,
    "overall": 0.775
  }
]
```

#### Scoring

Each object is scored with partial credit: `1 - |predicted - actual| / actual`, clamped to `[0, 1]`.

| Predicted | Actual | Score |
|-----------|--------|-------|
| 23 | 23 | 1.0 |
| 20 | 23 | 0.87 |
| 30 | 23 | 0.70 |
| 0 | 23 | 0.0 |

The `overall` score per image is the mean across all object types. The `__summary__` record averages each metric across all evaluated images.

---

## Table-reading workflow

### Stage 1 — Generate table images

```bash
uv run generate_tables.py --rows <R> --cols <C> --numbout <N> --format <fmt> [options]
```

Each cell of the grid contains a number rendered with randomised size, position, opacity, colour, and rotation.

#### Required flags

| Flag | Description |
|------|-------------|
| `--rows R` | Number of rows in the grid |
| `--cols C` | Number of columns in the grid |
| `--numbout N` | How many table images to generate |
| `--format fmt` | Output format: `png`, `jpg`, `jpeg`, or `gif` |

#### Optional flags

| Flag | Default | Description |
|------|---------|-------------|
| `--numbers LIST` | — | Comma-separated explicit values to place in cells (overrides `--digits`) |
| `--digits N` | `1` | Number of integer digits per cell (1 → 1–9, 2 → 10–99, etc.) |
| `--decimal` | off | Append a random single decimal digit to each number (e.g. `3.7`, `42.1`) |
| `--size-jitter F` | `0.2` | Font-size variation as a fraction of the base size |
| `--pos-jitter F` | `0.25` | Position jitter as a fraction of the cell size |
| `--opacity-min F` | `0.6` | Minimum opacity 0–1 |
| `--opacity-max F` | `1.0` | Maximum opacity 0–1 |
| `--rotation DEG` | `15` | Max rotation ± degrees |
| `--grid-color HEX` | `b0b0b0` | Gridline colour as hex |
| `--grid-width N` | `2` | Gridline width in pixels |
| `--canvas WxH` | `1200x900` | Canvas size in pixels |
| `--outdir PATH` | `output/tables_RxC_N_Ddigit[d]` | Output directory |
| `--seed N` | — | Integer seed for reproducible output |
| `--font PATH` | system font | Path to a TTF/OTF font file |

#### Output

The default directory name encodes the grid size, image count, digit count, and whether decimals are used. The current `config.toml` is also copied in for reproducibility.

```
output/tables_4x6_10_1digit/
├── table_0001.png
├── table_0002.png
├── ...
├── config.toml
└── validation.json
```

With `--digits 2 --decimal` the directory would be named `tables_4x6_10_2digitd`.

`validation.json` records the value of every cell as a 2-D grid:

```json
[
  {
    "filename": "table_0001.png",
    "rows": 4,
    "cols": 6,
    "grid": [
      ["2", "7", "1", "5", "3", "9"],
      ["4", "1", "8", "2", "6", "1"],
      ["9", "3", "5", "7", "2", "4"],
      ["1", "6", "3", "8", "5", "2"]
    ]
  }
]
```

#### Examples

```bash
# 4×6 grid of single digits, 10 images
uv run generate_tables.py --rows 4 --cols 6 --numbout 10 --format png

# 3×5 grid of two-digit integers
uv run generate_tables.py --rows 3 --cols 5 --numbout 10 --format png --digits 2

# Single digits with one decimal place (e.g. 3.7, 8.1)
uv run generate_tables.py --rows 4 --cols 6 --numbout 10 --format png --decimal

# Explicit value set, reproducible seed
uv run generate_tables.py --rows 3 --cols 4 --numbout 20 --format png \
    --numbers "1,2,5,10" --seed 42
```

---

### Stage 2 — Evaluate table reading

Add `table_dir` and `prompt_table` to `config.toml`:

```toml
[evaluation]
# ... existing composite_dir and prompt settings ...

# Directory produced by generate_tables.py
table_dir = "output/tables_4x6_10_1digit"

# Base prompt for evaluate_table.py.
# Grid dimensions and required JSON format are appended automatically.
prompt_table = """
Look at this table image carefully and read the number shown in every cell.
Return ONLY valid JSON — no explanation, no markdown.
"""
```

#### Run

```bash
uv run evaluate_table.py config.toml
```

#### Output

Results are written to `<table_dir>/score_table-<model-name>.json`. Three scores are reported per image:

| Score | Meaning |
|-------|---------|
| `cell_score` | Fraction of cells whose value was read correctly |
| `row_sum_score` | Fraction of row sums computed correctly |
| `col_sum_score` | Fraction of column sums computed correctly |
| `overall` | Mean of the three scores above |

```json
[
  {
    "filename": "table_0001.png",
    "actual_grid": [["2","7","1"],["4","1","8"]],
    "actual_row_sums": [10, 13],
    "actual_col_sums": [6, 8, 9],
    "cell_score": 0.83,
    "row_sum_score": 0.5,
    "col_sum_score": 0.67,
    "overall": 0.67
  },
  {
    "filename": "__summary__",
    "total_images": 10,
    "cell_score": 0.791,
    "row_sum_score": 0.6,
    "col_sum_score": 0.633,
    "overall": 0.675
  }
]
```

---

## Visualisation — plot_scores.py

Render a bar chart from all score files in a directory:

```bash
# Open an interactive window
uv run plot_scores.py output/tables_4x6_10_1digit

# Save to a file
uv run plot_scores.py output/tables_4x6_10_1digit --save chart.png
```

Works with both `score-*.json` (object-counting) and `score_table-*.json` (table-reading) files. Multiple score files in the same directory are shown side by side for model comparison.
