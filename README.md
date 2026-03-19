# generate_composites

Generates composite "counting" images by tiling randomly-selected instances of
source images onto a canvas.  Useful for creating image datasets where a model
or person must count the number of a particular object.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

```bash
# Install dependencies into a uv-managed environment
uv add pillow
```

Place your source images (PNG, JPEG, or GIF) in a folder called `sourceimg/`
next to the script:

```
project/
├── generate_composites.py
├── sourceimg/
│   ├── elephant.png
│   ├── giraffe.jpg
│   └── zebra.gif
└── output/          # created automatically
```

## Usage

```bash
uv run generate_composites.py --noobjects <N> --format <fmt> --numbout <N> [options]
```

### Required flags

| Flag | Description |
|------|-------------|
| `--noobjects N` | Total number of objects placed in each composite image |
| `--format fmt` | Output format: `png`, `jpg`, `jpeg`, `gif`, or `pdf` |
| `--numbout N` | How many composite images to generate |

### Optional flags

| Flag | Default | Description |
|------|---------|-------------|
| `--layout` | `grid` | Placement style: `grid`, `random`, or `spiral` |
| `--target name` | — | Fix one image type as the "target" (stem name, e.g. `elephant`). Must be used with `--numtarget`. |
| `--numtarget N` | — | Number of target instances fixed in every composite. The remaining `noobjects - numtarget` slots are filled randomly from the other images. Must be used with `--target`. |
| `--sourcedir path` | `sourceimg` | Directory containing source images |
| `--canvas WxH` | `1200x900` | Canvas size in pixels |
| `--seed N` | — | Integer seed for reproducible output |

### Layout modes

| Mode | Behaviour |
|------|-----------|
| `grid` | Objects arranged in a regular grid with slight random jitter |
| `random` | Objects scattered at random positions; placement retries to minimise overlap |
| `spiral` | Objects follow an Archimedean spiral outward from the canvas centre |

## Output

Images are written to:

```
output/<format>_<numbout>_<noobjects>/composite_0001.<ext>
output/<format>_<numbout>_<noobjects>/composite_0002.<ext>
...
```

## Examples

Generate 10 PNG composites each containing 4 objects, grid layout:

```bash
uv run generate_composites.py --noobjects 4 --format png --numbout 10
```

Generate 20 composites with a spiral layout and a larger canvas:

```bash
uv run generate_composites.py --noobjects 8 --format jpg --numbout 20 \
    --layout spiral --canvas 1600x1200
```

Fix 8 elephants in every composite, fill the remaining 12 slots randomly,
reproducible with a seed:

```bash
uv run generate_composites.py --noobjects 20 --format png --numbout 50 \
    --target elephant --numtarget 8 --seed 42
```

Generate PDFs with a random scatter layout:

```bash
uv run generate_composites.py --noobjects 6 --format pdf --numbout 5 \
    --layout random
```
