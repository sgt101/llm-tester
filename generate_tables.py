#!/usr/bin/env python3
"""
Generate composite images containing numbers arranged in a table/grid layout.

Each cell of the table contains one number rendered with randomised size,
position, opacity, colour, and rotation.  The resulting images and a
validation.json ground-truth file can be fed directly into evaluate.py.

Usage:
    uv run generate_tables.py --rows R --cols C --numbout N --format fmt [options]

Required flags:
    --rows R        Number of rows in the grid
    --cols C        Number of columns in the grid
    --numbout N     Number of composite images to generate
    --format fmt    Output format: png, jpg, jpeg, or gif

Optional flags:
    --numbers LIST    Comma-separated values to place in cells (default: 1,2,3,4,5,6,7,8,9)
    --canvas WxH      Canvas size in pixels (default: 1200x900)
    --size-jitter F   Font-size variation as a fraction of the base size (default: 0.2)
    --pos-jitter F    Position jitter as a fraction of the cell size (default: 0.25)
    --opacity-min F   Minimum opacity 0.0–1.0 (default: 0.6)
    --opacity-max F   Maximum opacity 0.0–1.0 (default: 1.0)
    --rotation F      Max rotation in degrees, applied ± symmetrically (default: 15)
    --grid-color HEX  Gridline colour as a hex string, e.g. 808080 (default: b0b0b0)
    --grid-width N    Gridline width in pixels (default: 2)
    --outdir PATH     Output directory (default: output/tables_RxC_N)
    --seed N          Integer seed for reproducible output
    --font PATH       Path to a TTF/OTF font file (uses a system font if omitted)
"""

import argparse
import colorsys
import json
import random
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: Pillow is required. Install with: uv add pillow")
    sys.exit(1)


SUPPORTED_FORMATS = {"png", "jpg", "jpeg", "gif"}

# Candidate system fonts tried in order when --font is not supplied
_SYSTEM_FONT_CANDIDATES = [
    # macOS
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]


def _load_font(size: int, font_path: str | None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path:
        return ImageFont.truetype(font_path, size)
    for candidate in _SYSTEM_FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                continue
    # Last resort: PIL built-in bitmap font (ignores size)
    return ImageFont.load_default()


def _random_color(rng: random.Random) -> tuple[int, int, int]:
    """Return a vivid, readable colour on a white background."""
    h = rng.random()
    s = rng.uniform(0.55, 1.0)
    v = rng.uniform(0.25, 0.75)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    if len(hex_str) != 6:
        raise ValueError(f"Expected a 6-digit hex colour, got '{hex_str}'")
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return r, g, b


def _render_cell(
    text: str,
    cell_w: int,
    cell_h: int,
    base_font_size: int,
    size_jitter: float,
    pos_jitter: float,
    opacity_min: float,
    opacity_max: float,
    rotation: float,
    rng: random.Random,
    font_path: str | None,
) -> Image.Image:
    """Render *text* onto a transparent RGBA image sized (cell_w, cell_h)."""

    # --- Randomise per-cell parameters ---
    font_size = max(8, int(base_font_size * (1.0 + rng.uniform(-size_jitter, size_jitter))))
    opacity   = int(rng.uniform(opacity_min, opacity_max) * 255)
    angle     = rng.uniform(-rotation, rotation)
    color     = _random_color(rng)

    font = _load_font(font_size, font_path)

    # Render onto a padded canvas so rotation never clips the glyph
    pad = int(max(cell_w, cell_h) * 0.6)
    canvas_size = (cell_w + pad * 2, cell_h + pad * 2)
    tmp = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(tmp)

    # Measure text
    bbox   = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Centre of cell within padded canvas, with jitter
    jitter_x = rng.uniform(-pos_jitter, pos_jitter) * cell_w
    jitter_y = rng.uniform(-pos_jitter, pos_jitter) * cell_h
    cx = pad + cell_w / 2 + jitter_x
    cy = pad + cell_h / 2 + jitter_y

    tx = cx - text_w / 2 - bbox[0]
    ty = cy - text_h / 2 - bbox[1]

    draw.text((tx, ty), text, font=font, fill=(*color, opacity))

    # Rotate around the nominal centre of the number
    tmp = tmp.rotate(angle, center=(cx, cy), resample=Image.BICUBIC)

    # Crop back to cell dimensions
    return tmp.crop((pad, pad, pad + cell_w, pad + cell_h))


def _random_number(
    rng: random.Random,
    numbers: list[str] | None,
    digits: int,
    decimal: bool,
) -> str:
    """Generate one cell value according to the active mode."""
    if numbers is not None:
        integer_part = rng.choice(numbers)
    else:
        lo = 10 ** (digits - 1) if digits > 1 else 1
        hi = 10 ** digits - 1
        integer_part = str(rng.randint(lo, hi))
    if decimal:
        return f"{integer_part}.{rng.randint(0, 9)}"
    return integer_part


def generate_table(
    numbers: list[str] | None,
    rows: int,
    cols: int,
    canvas_w: int,
    canvas_h: int,
    base_font_size: int,
    size_jitter: float,
    pos_jitter: float,
    opacity_min: float,
    opacity_max: float,
    rotation: float,
    rng: random.Random,
    font_path: str | None,
    grid_color: tuple[int, int, int],
    grid_width: int,
    digits: int = 1,
    decimal: bool = False,
) -> tuple[Image.Image, list[list[str]]]:
    """
    Build one table image.

    Returns (image, grid) where grid is a 2-D list of the value placed in
    each cell: grid[row][col].
    """
    margin = 20
    grid_w  = canvas_w - 2 * margin
    grid_h  = canvas_h - 2 * margin
    cell_w  = grid_w // cols
    cell_h  = grid_h // rows

    img  = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    grid: list[list[str]] = []

    for row in range(rows):
        grid_row: list[str] = []
        for col in range(cols):
            number = _random_number(rng, numbers, digits, decimal)
            grid_row.append(number)

            x0 = margin + col * cell_w
            y0 = margin + row * cell_h

            cell_img = _render_cell(
                number, cell_w, cell_h, base_font_size,
                size_jitter, pos_jitter, opacity_min, opacity_max,
                rotation, rng, font_path,
            )
            img.paste(cell_img, (x0, y0), mask=cell_img)
        grid.append(grid_row)

    # Draw gridlines on top so they're always visible
    for c in range(cols + 1):
        x = margin + c * cell_w
        draw.line([(x, margin), (x, margin + rows * cell_h)],
                  fill=(*grid_color, 255), width=grid_width)
    for r in range(rows + 1):
        y = margin + r * cell_h
        draw.line([(margin, y), (margin + cols * cell_w, y)],
                  fill=(*grid_color, 255), width=grid_width)

    return img, grid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate table/grid composite images with numbers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--rows",        type=int, required=True, help="Number of grid rows")
    parser.add_argument("--cols",        type=int, required=True, help="Number of grid columns")
    parser.add_argument("--numbout",     type=int, required=True, help="Number of images to generate")
    parser.add_argument("--format",      required=True, choices=sorted(SUPPORTED_FORMATS), help="Output image format")
    parser.add_argument("--numbers",     default=None, help="Comma-separated explicit values to use (overrides --digits)")
    parser.add_argument("--digits",      type=int, default=1, metavar="N", help="Number of integer digits per cell (default: 1, i.e. 1–9)")
    parser.add_argument("--decimal",     action="store_true", help="Append a random single decimal digit to each number")
    parser.add_argument("--canvas",      default="1200x900", help="Canvas size WxH in pixels")
    parser.add_argument("--size-jitter", type=float, default=0.2,  dest="size_jitter",  metavar="F", help="Font-size variation fraction (default: 0.2)")
    parser.add_argument("--pos-jitter",  type=float, default=0.25, dest="pos_jitter",   metavar="F", help="Position jitter fraction (default: 0.25)")
    parser.add_argument("--opacity-min", type=float, default=0.6,  dest="opacity_min",  metavar="F", help="Minimum opacity 0–1 (default: 0.6)")
    parser.add_argument("--opacity-max", type=float, default=1.0,  dest="opacity_max",  metavar="F", help="Maximum opacity 0–1 (default: 1.0)")
    parser.add_argument("--rotation",    type=float, default=15.0, metavar="DEG",        help="Max rotation ± degrees (default: 15)")
    parser.add_argument("--grid-color",  default="b0b0b0",         dest="grid_color",   metavar="HEX", help="Gridline colour hex (default: b0b0b0)")
    parser.add_argument("--grid-width",  type=int, default=2,      dest="grid_width",   metavar="N",   help="Gridline width in pixels (default: 2)")
    parser.add_argument("--outdir",      default=None,             help="Output directory")
    parser.add_argument("--seed",        type=int, default=None,   help="Random seed")
    parser.add_argument("--font",        default=None,             help="Path to TTF/OTF font file")
    args = parser.parse_args()

    # --- Validate canvas ---
    try:
        canvas_w, canvas_h = (int(v) for v in args.canvas.lower().split("x"))
    except ValueError:
        parser.error("--canvas must be WxH, e.g. 1200x900")

    # --- Validate numbers / digits ---
    if args.numbers is not None:
        numbers: list[str] | None = [n.strip() for n in args.numbers.split(",") if n.strip()]
        if not numbers:
            parser.error("--numbers must be a non-empty comma-separated list")
    else:
        numbers = None
        if args.digits < 1:
            parser.error("--digits must be at least 1")

    # --- Validate opacity ---
    if not (0.0 <= args.opacity_min <= args.opacity_max <= 1.0):
        parser.error("opacity values must satisfy 0 ≤ opacity-min ≤ opacity-max ≤ 1")

    # --- Validate grid colour ---
    try:
        grid_color = _hex_to_rgb(args.grid_color)
    except ValueError as e:
        parser.error(f"--grid-color: {e}")

    fmt = args.format.lower()

    # --- Output directory ---
    if args.outdir:
        out_dir = Path(args.outdir)
    else:
        decimal_suffix = "d" if args.decimal else ""
        out_dir = Path("output") / f"tables_{args.rows}x{args.cols}_{args.numbout}_{args.digits}digit{decimal_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Base font size: ~60 % of the smaller cell dimension
    cell_w = (canvas_w - 40) // args.cols
    cell_h = (canvas_h - 40) // args.rows
    base_font_size = int(min(cell_w, cell_h) * 0.6)

    rng = random.Random(args.seed)

    number_desc = (
        f"explicit list: {', '.join(numbers)}" if numbers is not None
        else f"{args.digits}-digit{'  +decimal' if args.decimal else ''}"
    )
    print(f"Grid     : {args.rows} rows × {args.cols} cols  ({args.rows * args.cols} cells per image)")
    print(f"Numbers  : {number_desc}")
    print(f"Images   : {args.numbout}  →  {out_dir}\n")

    records: list[dict] = []

    for i in range(args.numbout):
        fname = f"table_{i + 1:04d}.{fmt}"
        img, grid = generate_table(
            numbers=numbers,
            rows=args.rows,
            cols=args.cols,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            base_font_size=base_font_size,
            size_jitter=args.size_jitter,
            pos_jitter=args.pos_jitter,
            opacity_min=args.opacity_min,
            opacity_max=args.opacity_max,
            rotation=args.rotation,
            rng=rng,
            font_path=args.font,
            grid_color=grid_color,
            grid_width=args.grid_width,
            digits=args.digits,
            decimal=args.decimal,
        )

        out_path = out_dir / fname
        if fmt in ("jpg", "jpeg"):
            img = img.convert("RGB")
            img.save(out_path, optimize=True, quality=92)
        else:
            img.save(out_path)

        record: dict = {"filename": fname, "rows": args.rows, "cols": args.cols, "grid": grid}
        records.append(record)

        print(f"  [{i + 1:>{len(str(args.numbout))}}/{args.numbout}] {fname}")

    validation_path = out_dir / "validation.json"
    validation_path.write_text(json.dumps(records, indent=2))

    config_src = Path("config.toml")
    if config_src.exists():
        import shutil
        shutil.copy2(config_src, out_dir / "config.toml")
        print(f"\nconfig.toml copied to {out_dir}")

    print(f"validation.json written to {out_dir}")


if __name__ == "__main__":
    main()
