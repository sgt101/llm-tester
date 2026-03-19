#!/usr/bin/env python3
"""
Generate composite counting images from source images.

Usage:
    python generate_composites.py --noobjects 4 --format png --numbout 10
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif"}


def load_source_images(source_dir: str) -> list[tuple[str, Image.Image]]:
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"Error: Source directory '{source_dir}' not found.")
        sys.exit(1)

    images = []
    for file in sorted(source_path.iterdir()):
        if file.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            img = Image.open(file)
            # For animated GIFs, take only the first frame
            if hasattr(img, "n_frames") and img.n_frames > 1:
                img.seek(0)
            img = img.copy()
            images.append((file.stem, img))
            print(f"  Loaded: {file.name}")
        except Exception as e:
            print(f"  Warning: Could not load {file.name}: {e}")

    if not images:
        print(f"Error: No valid images found in '{source_dir}'.")
        sys.exit(1)

    return images


def random_distribution(num_types: int, total: int) -> list[int]:
    """Return a list of counts per type that sum to total, each >= 0, at least one > 0."""
    if num_types == 0 or total == 0:
        return [0] * num_types

    if num_types >= total:
        # Pick `total` distinct types, each gets 1
        chosen = random.sample(range(num_types), total)
        dist = [0] * num_types
        for i in chosen:
            dist[i] = 1
        return dist

    # Each type gets at least 1, distribute the remainder
    dist = [1] * num_types
    for _ in range(total - num_types):
        dist[random.randrange(num_types)] += 1
    random.shuffle(dist)
    return dist


def distribution_with_target(
    num_types: int, total: int, target_idx: int, target_count: int
) -> list[int]:
    """Return a distribution where index target_idx is fixed at target_count.

    The remaining (total - target_count) objects are distributed randomly
    among the non-target types.
    """
    remainder = total - target_count
    non_target_count = num_types - 1

    if non_target_count == 0:
        # Only one image type and it IS the target
        dist = [0] * num_types
        dist[target_idx] = target_count
        return dist

    # Get a distribution for the non-target slots
    non_target_dist = random_distribution(non_target_count, remainder)

    # Splice the target back in at the correct index
    dist: list[int] = []
    nt_iter = iter(non_target_dist)
    for i in range(num_types):
        if i == target_idx:
            dist.append(target_count)
        else:
            dist.append(next(nt_iter))
    return dist


def _prepare_instances(
    images: list[tuple[str, Image.Image]],
    distribution: list[int],
    obj_w: int,
    obj_h: int,
) -> list[Image.Image]:
    """Build and shuffle the list of thumbnailed instances."""
    instances: list[Image.Image] = []
    for (_, img), count in zip(images, distribution):
        for _ in range(count):
            thumb = img.copy()
            if thumb.mode not in ("RGB", "RGBA", "L"):
                thumb = thumb.convert("RGBA")
            thumb.thumbnail((obj_w, obj_h), Image.LANCZOS)
            instances.append(thumb)
    random.shuffle(instances)
    return instances


def _paste(canvas: Image.Image, thumb: Image.Image, x: int, y: int) -> None:
    if thumb.mode == "RGBA":
        canvas.paste(thumb, (x, y), thumb)
    else:
        canvas.paste(thumb.convert("RGB"), (x, y))


def _clamp(x: int, y: int, tw: int, th: int, cw: int, ch: int) -> tuple[int, int]:
    return max(0, min(x, cw - tw)), max(0, min(y, ch - th))


# ---------------------------------------------------------------------------
# Layout: grid
# ---------------------------------------------------------------------------

def _layout_grid(
    instances: list[Image.Image],
    canvas: Image.Image,
) -> None:
    """Regular grid with slight random jitter."""
    total = len(instances)
    cw, ch = canvas.size
    cols = math.ceil(math.sqrt(total))
    rows = math.ceil(total / cols)
    cell_w = cw // cols
    cell_h = ch // rows

    for idx, thumb in enumerate(instances):
        row = idx // cols
        col = idx % cols
        jitter_x = random.randint(-cell_w // 10, cell_w // 10)
        jitter_y = random.randint(-cell_h // 10, cell_h // 10)
        x = col * cell_w + (cell_w - thumb.width) // 2 + jitter_x
        y = row * cell_h + (cell_h - thumb.height) // 2 + jitter_y
        x, y = _clamp(x, y, thumb.width, thumb.height, cw, ch)
        _paste(canvas, thumb, x, y)


# ---------------------------------------------------------------------------
# Layout: random (scatter, attempts to avoid overlap)
# ---------------------------------------------------------------------------

def _layout_random(
    instances: list[Image.Image],
    canvas: Image.Image,
) -> None:
    """Place each object at a random position, retrying to reduce overlap."""
    cw, ch = canvas.size
    placed: list[tuple[int, int, int, int]] = []  # (x, y, w, h)
    max_attempts = 60

    for thumb in instances:
        tw, th = thumb.width, thumb.height
        best = None
        best_overlap = float("inf")

        for _ in range(max_attempts):
            x = random.randint(0, max(0, cw - tw))
            y = random.randint(0, max(0, ch - th))
            # Calculate total overlap with already-placed objects
            overlap = 0
            for px, py, pw, ph in placed:
                ox = max(0, min(x + tw, px + pw) - max(x, px))
                oy = max(0, min(y + th, py + ph) - max(y, py))
                overlap += ox * oy
            if overlap == 0:
                best = (x, y)
                break
            if overlap < best_overlap:
                best_overlap = overlap
                best = (x, y)

        x, y = best  # type: ignore[misc]
        placed.append((x, y, tw, th))
        _paste(canvas, thumb, x, y)


# ---------------------------------------------------------------------------
# Layout: spiral
# ---------------------------------------------------------------------------

def _layout_spiral(
    instances: list[Image.Image],
    canvas: Image.Image,
) -> None:
    """Place objects along an Archimedean spiral from the canvas centre."""
    total = len(instances)
    cw, ch = canvas.size
    cx, cy = cw / 2, ch / 2

    # Scale the spiral so the outermost ring stays inside the canvas
    max_radius = min(cx, cy) * 0.88
    # Spacing between successive spiral arms based on the largest thumbnail
    max_thumb = max((max(t.width, t.height) for t in instances), default=80)
    arm_spacing = max_thumb * 1.1

    for idx, thumb in enumerate(instances):
        tw, th = thumb.width, thumb.height
        if total == 1:
            angle = 0.0
            radius = 0.0
        else:
            # Distribute evenly by arc-length: angle grows so that
            # r = arm_spacing * angle / (2π), arc ≈ arm_spacing per step
            angle = math.sqrt(2 * arm_spacing * idx * (2 * math.pi) / arm_spacing)
            radius = arm_spacing * angle / (2 * math.pi)
            # If the spiral would exceed max_radius, wrap back to centre
            if radius > max_radius:
                angle = math.sqrt(2 * arm_spacing * (idx % max(1, total // 2)) * (2 * math.pi) / arm_spacing)
                radius = arm_spacing * angle / (2 * math.pi)

        x = int(cx + radius * math.cos(angle) - tw / 2)
        y = int(cy + radius * math.sin(angle) - th / 2)
        x, y = _clamp(x, y, tw, th, cw, ch)
        _paste(canvas, thumb, x, y)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

LAYOUTS = {"grid": _layout_grid, "random": _layout_random, "spiral": _layout_spiral}


def create_composite(
    images: list[tuple[str, Image.Image]],
    distribution: list[int],
    canvas_size: tuple[int, int] = (1200, 900),
    bg_color: tuple[int, int, int] = (255, 255, 255),
    layout: str = "grid",
) -> Image.Image:
    """Arrange object instances on a canvas using the chosen layout."""
    total = sum(distribution)
    canvas = Image.new("RGB", canvas_size, bg_color)
    if total == 0:
        return canvas

    # Compute a uniform object size that fits comfortably for the grid layout.
    # For random/spiral we use a fixed size so objects don't shrink unnecessarily.
    cols = math.ceil(math.sqrt(total))
    rows = math.ceil(total / cols)
    cell_w = canvas_size[0] // cols
    cell_h = canvas_size[1] // rows
    padding = max(4, min(cell_w, cell_h) // 8)
    grid_obj_w = max(1, cell_w - 2 * padding)
    grid_obj_h = max(1, cell_h - 2 * padding)

    if layout == "grid":
        obj_w, obj_h = grid_obj_w, grid_obj_h
    else:
        # For random/spiral: target ~15% of canvas in the smaller dimension,
        # but never larger than the grid cell size (which still shrinks for
        # very high counts so objects remain distinguishable).
        free_obj = max(canvas_size[0], canvas_size[1]) // 6
        obj_w = min(free_obj, grid_obj_w * 3)
        obj_h = min(free_obj, grid_obj_h * 3)
        obj_w = max(1, obj_w)
        obj_h = max(1, obj_h)

    instances = _prepare_instances(images, distribution, obj_w, obj_h)

    layout_fn = LAYOUTS.get(layout)
    if layout_fn is None:
        raise ValueError(f"Unknown layout '{layout}'. Choose from: {', '.join(LAYOUTS)}")
    layout_fn(instances, canvas)

    return canvas


def save_composite(img: Image.Image, path: Path, fmt: str) -> None:
    fmt_lower = fmt.lower()
    if fmt_lower in ("jpg", "jpeg"):
        canvas = img.convert("RGB")
        canvas.save(path, "JPEG", quality=92)
    elif fmt_lower == "png":
        img.save(path, "PNG")
    elif fmt_lower == "gif":
        img.convert("RGB").save(path, "GIF")
    elif fmt_lower == "pdf":
        img.convert("RGB").save(path, "PDF", resolution=150)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate composite counting images from a set of source images."
    )
    parser.add_argument(
        "--noobjects",
        type=int,
        required=True,
        help="Total number of objects to place in each composite image",
    )
    parser.add_argument(
        "--format",
        type=str,
        required=True,
        choices=["jpg", "jpeg", "gif", "png", "pdf"],
        help="Output image format",
    )
    parser.add_argument(
        "--numbout",
        type=int,
        required=True,
        help="Number of composite images to generate",
    )
    parser.add_argument(
        "--sourcedir",
        type=str,
        default="sourceimg",
        help="Directory containing source images (default: sourceimg)",
    )
    parser.add_argument(
        "--canvas",
        type=str,
        default="1200x900",
        help="Canvas size as WxH in pixels (default: 1200x900)",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="grid",
        choices=["grid", "random", "spiral"],
        help="Object placement layout: grid (default), random, or spiral",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Stem name of the target image (e.g. 'elephant'). Requires --numtarget.",
    )
    parser.add_argument(
        "--numtarget",
        type=int,
        default=None,
        help="Fixed number of target instances in every composite. Requires --target.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    try:
        canvas_w, canvas_h = (int(v) for v in args.canvas.lower().split("x"))
    except ValueError:
        print("Error: --canvas must be in WxH format, e.g. 1200x900")
        sys.exit(1)

    # Normalise format: treat jpeg as jpg for directory/file naming
    fmt = args.format.lower()
    file_ext = "jpg" if fmt == "jpeg" else fmt

    # Output directory: output/{format}_{numbout}_{noobjects}/
    out_dir = Path("output") / f"{file_ext}_{args.numbout}_{args.noobjects}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate --target / --numtarget pairing
    if (args.target is None) != (args.numtarget is None):
        parser.error("--target and --numtarget must be used together.")

    print(f"Loading source images from '{args.sourcedir}'...")
    images = load_source_images(args.sourcedir)
    print(f"Found {len(images)} source image(s).\n")

    # Resolve target index after images are loaded
    target_idx: int | None = None
    if args.target is not None:
        names = [name for name, _ in images]
        matches = [i for i, n in enumerate(names) if n.lower() == args.target.lower()]
        if not matches:
            print(f"Error: --target '{args.target}' not found. Available: {', '.join(names)}")
            sys.exit(1)
        target_idx = matches[0]
        if args.numtarget < 0:
            parser.error("--numtarget must be >= 0.")
        if args.numtarget > args.noobjects:
            parser.error(f"--numtarget ({args.numtarget}) cannot exceed --noobjects ({args.noobjects}).")
        print(f"Target: '{args.target}' fixed at {args.numtarget} instance(s) per composite.")

    print(f"Generating {args.numbout} composite(s) with {args.noobjects} object(s) each.")
    print(f"Output directory: {out_dir}\n")

    validation: list[dict] = []
    all_names = [name for name, _ in images]

    for i in range(1, args.numbout + 1):
        if target_idx is not None:
            dist = distribution_with_target(len(images), args.noobjects, target_idx, args.numtarget)
        else:
            dist = random_distribution(len(images), args.noobjects)
        desc = ", ".join(
            f"{cnt}x {name}"
            for (name, _), cnt in zip(images, dist)
            if cnt > 0
        )
        print(f"  [{i:>{len(str(args.numbout))}}/{args.numbout}] {desc}")

        composite = create_composite(images, dist, canvas_size=(canvas_w, canvas_h), layout=args.layout)

        fname = f"composite_{i:04d}.{file_ext}"
        save_composite(composite, out_dir / fname, fmt)

        record: dict = {"filename": fname}
        record.update({name: cnt for name, cnt in zip(all_names, dist)})
        validation.append(record)

    validation_path = out_dir / "validation.json"
    validation_path.write_text(json.dumps(validation, indent=2))
    print(f"\nDone. {args.numbout} image(s) saved to: {out_dir}")
    print(f"Validation file: {validation_path}")


if __name__ == "__main__":
    main()
