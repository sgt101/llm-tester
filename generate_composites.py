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
) -> list[tuple[Image.Image, int]]:
    """Build and shuffle the list of (thumbnail, type_index) pairs."""
    instances: list[tuple[Image.Image, int]] = []
    for type_idx, ((_, img), count) in enumerate(zip(images, distribution)):
        for _ in range(count):
            thumb = img.copy()
            if thumb.mode not in ("RGB", "RGBA", "L"):
                thumb = thumb.convert("RGBA")
            thumb.thumbnail((obj_w, obj_h), Image.LANCZOS)
            instances.append((thumb, type_idx))
    random.shuffle(instances)
    return instances


def _paste(canvas: Image.Image, thumb: Image.Image, x: int, y: int) -> None:
    if thumb.mode == "RGBA":
        canvas.paste(thumb, (x, y), thumb)
    else:
        canvas.paste(thumb.convert("RGB"), (x, y))


def _clamp(x: int, y: int, tw: int, th: int, cw: int, ch: int) -> tuple[int, int]:
    return max(0, min(x, cw - tw)), max(0, min(y, ch - th))


def _overlaps(
    x: int, y: int, tw: int, th: int,
    placed: list[tuple[int, int, int, int]],
    margin: int = 6,
) -> bool:
    """Return True if the rectangle (x, y, tw, th) overlaps any placed rectangle."""
    for px, py, pw, ph in placed:
        if (x < px + pw + margin and x + tw + margin > px and
                y < py + ph + margin and y + th + margin > py):
            return True
    return False


_SHRINK_STEP = 0.15   # scale reduction per retry pass
_MIN_SCALE   = 0.25   # never shrink below 25 % of original size


def _shrink(thumb: Image.Image, scale: float) -> Image.Image:
    w = max(1, int(thumb.width  * scale))
    h = max(1, int(thumb.height * scale))
    return thumb.resize((w, h), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Layout: grid
# ---------------------------------------------------------------------------

def _layout_grid(
    instances: list[tuple[Image.Image, int]],
    canvas: Image.Image,
) -> list[int]:
    """Regular grid with slight random jitter. Always places every instance."""
    total = len(instances)
    cw, ch = canvas.size
    cols = math.ceil(math.sqrt(total))
    rows = math.ceil(total / cols)
    cell_w = cw // cols
    cell_h = ch // rows
    placed_types: list[int] = []

    for idx, (thumb, type_idx) in enumerate(instances):
        row = idx // cols
        col = idx % cols
        jitter_x = random.randint(-cell_w // 10, cell_w // 10)
        jitter_y = random.randint(-cell_h // 10, cell_h // 10)
        x = col * cell_w + (cell_w - thumb.width) // 2 + jitter_x
        y = row * cell_h + (cell_h - thumb.height) // 2 + jitter_y
        x, y = _clamp(x, y, thumb.width, thumb.height, cw, ch)
        _paste(canvas, thumb, x, y)
        placed_types.append(type_idx)

    return placed_types


# ---------------------------------------------------------------------------
# Layout: random (scatter, guaranteed non-overlapping)
# ---------------------------------------------------------------------------

def _layout_random(
    instances: list[tuple[Image.Image, int]],
    canvas: Image.Image,
) -> list[int]:
    """Place each object at a random non-overlapping position.

    For each object, up to 500 random positions are tried.  If none are
    clear, the object is shrunk by _SHRINK_STEP and the search retries
    until _MIN_SCALE is reached.  Returns type indices of placed objects.
    """
    cw, ch = canvas.size
    placed: list[tuple[int, int, int, int]] = []
    placed_types: list[int] = []

    for thumb, type_idx in instances:
        placed_this = False
        scale = 1.0

        while not placed_this and scale >= _MIN_SCALE:
            current = thumb if scale == 1.0 else _shrink(thumb, scale)
            tw, th = current.width, current.height

            for _ in range(500):
                x = random.randint(0, max(0, cw - tw))
                y = random.randint(0, max(0, ch - th))
                if not _overlaps(x, y, tw, th, placed):
                    placed.append((x, y, tw, th))
                    _paste(canvas, current, x, y)
                    placed_this = True
                    placed_types.append(type_idx)
                    break

            if not placed_this:
                scale -= _SHRINK_STEP

        if not placed_this:
            print("\n  Warning: could not place one object even at minimum size — skipped.")

    return placed_types


# ---------------------------------------------------------------------------
# Layout: spiral (arc-walking, guaranteed non-overlapping)
# ---------------------------------------------------------------------------

def _layout_spiral(
    instances: list[tuple[Image.Image, int]],
    canvas: Image.Image,
) -> list[int]:
    """Place objects along an Archimedean spiral, advancing the arc until
    each new object sits clear of all previously placed ones.

    If the spiral is exhausted at the current size the object is shrunk by
    _SHRINK_STEP and the spiral restarts from angle 0 until _MIN_SCALE is
    reached.  Returns type indices of placed objects.
    """
    cw, ch = canvas.size
    cx, cy = cw / 2, ch / 2
    max_radius = min(cx, cy) * 0.90
    delta_angle = 0.08  # radians (~4.6°)

    placed: list[tuple[int, int, int, int]] = []
    placed_types: list[int] = []
    angle = 0.0

    for thumb, type_idx in instances:
        placed_this = False
        scale = 1.0

        while not placed_this and scale >= _MIN_SCALE:
            current = thumb if scale == 1.0 else _shrink(thumb, scale)
            tw, th = current.width, current.height

            # Arm spacing scales with the current object size so shrunken
            # objects still pack tightly rather than leaving huge gaps.
            arm_spacing = max(tw, th) * 1.2

            search_angle = angle

            while True:
                radius = arm_spacing * search_angle / (2 * math.pi)
                if radius > max_radius:
                    scale -= _SHRINK_STEP
                    break

                x = int(cx + radius * math.cos(search_angle) - tw / 2)
                y = int(cy + radius * math.sin(search_angle) - th / 2)
                x, y = _clamp(x, y, tw, th, cw, ch)

                if not _overlaps(x, y, tw, th, placed):
                    placed.append((x, y, tw, th))
                    _paste(canvas, current, x, y)
                    placed_this = True
                    placed_types.append(type_idx)
                    r = max(arm_spacing * 0.5, radius)
                    angle = search_angle + max(delta_angle, max(tw, th) / r)
                    break

                search_angle += delta_angle

        if not placed_this:
            print("\n  Warning: could not place one object even at minimum size — skipped.")

    return placed_types


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
) -> tuple[Image.Image, list[int]]:
    """Arrange object instances on a canvas using the chosen layout.

    Returns (canvas, actual_distribution) where actual_distribution reflects
    only the objects that were successfully placed — which may be fewer than
    the requested distribution if placement failed for some objects.
    """
    total = sum(distribution)
    canvas = Image.new("RGB", canvas_size, bg_color)
    if total == 0:
        return canvas, [0] * len(distribution)

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
        free_obj = max(canvas_size[0], canvas_size[1]) // 6
        obj_w = min(free_obj, grid_obj_w * 3)
        obj_h = min(free_obj, grid_obj_h * 3)
        obj_w = max(1, obj_w)
        obj_h = max(1, obj_h)

    instances = _prepare_instances(images, distribution, obj_w, obj_h)

    layout_fn = LAYOUTS.get(layout)
    if layout_fn is None:
        raise ValueError(f"Unknown layout '{layout}'. Choose from: {', '.join(LAYOUTS)}")
    placed_type_indices = layout_fn(instances, canvas)

    # Reconstruct actual distribution from what was placed.
    actual_dist = [0] * len(distribution)
    for type_idx in placed_type_indices:
        actual_dist[type_idx] += 1

    return canvas, actual_dist


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
    target_suffix = f"_target{args.numtarget}" if args.numtarget is not None else ""
    out_dir = Path("output") / f"{file_ext}_{args.numbout}_{args.noobjects}_{args.layout}{target_suffix}"
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

        composite, actual_dist = create_composite(images, dist, canvas_size=(canvas_w, canvas_h), layout=args.layout)

        fname = f"composite_{i:04d}.{file_ext}"
        save_composite(composite, out_dir / fname, fmt)

        record: dict = {"filename": fname}
        record.update({name: cnt for name, cnt in zip(all_names, actual_dist)})
        validation.append(record)

    validation_path = out_dir / "validation.json"
    validation_path.write_text(json.dumps(validation, indent=2))

    config_src = Path("config.toml")
    if config_src.exists():
        import shutil
        shutil.copy2(config_src, out_dir / "config.toml")
        print(f"\nconfig.toml copied to {out_dir}")

    print(f"\nDone. {args.numbout} image(s) saved to: {out_dir}")
    print(f"Validation file: {validation_path}")


if __name__ == "__main__":
    main()
