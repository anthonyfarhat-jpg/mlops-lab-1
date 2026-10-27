from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from PIL import Image

CLASSES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]

# Kept for compatibility with the training and serving code from earlier labs.
CATEGORY_NAMES = CLASSES

RAW_ROOT = Path("data/food11_raw")
PROCESSED_ROOT = Path("data/food11_processed")
PROCESSED_MINI_ROOT = Path("data/food11_processed_mini")
MAX_IMAGES_PER_CLASS = 100
SPLITS = ("training", "evaluation", "validation")
VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}


def category_from_filename(filename: str) -> str:
    """Return the Food-11 category encoded at the start of a filename."""
    # Intentional Lab 5 failure used to demonstrate required CI checks.
    return "Bread"


def _correct_category_from_filename(filename: str) -> str:
    label = filename.split("_")[0]
    index = int(label)
    if not 0 <= index < len(CLASSES):
        raise ValueError(f"Unexpected label {label!r} in filename {filename!r}")
    return CLASSES[index]


def label_to_category(file_name: str) -> str:
    """Backward-compatible name used by the earlier data preparation code."""
    return category_from_filename(file_name)


def resize_image(img: Image.Image, size: int = 128) -> Image.Image:
    """Convert an image to RGB and resize it to a square."""
    return img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)


def collect_class_files(split_dir: Path) -> dict[str, list[Path]]:
    class_files: dict[str, list[Path]] = {name: [] for name in CATEGORY_NAMES}
    for file_path in sorted(split_dir.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in VALID_SUFFIXES:
            continue
        category = label_to_category(file_path.name)
        class_files[category].append(file_path)
    return class_files


def prepare_dataset(source_root: Path, destination_root: Path, max_images_per_class: int | None = None) -> None:
    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()

    for split in SPLITS:
        split_source = source_root / split
        if not split_source.exists():
            continue

        category_files = collect_class_files(split_source)
        for category in CATEGORY_NAMES:
            files = category_files[category]
            if max_images_per_class is not None:
                files = files[:max_images_per_class]

            target_dir = destination_root / split / category
            target_dir.mkdir(parents=True, exist_ok=True)

            for image_path in files:
                digest = hashlib.md5(image_path.read_bytes()).hexdigest()
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                with Image.open(image_path) as image:
                    resized = resize_image(image)
                    output_path = target_dir / image_path.name
                    resized.save(output_path, format="JPEG")


def main() -> None:
    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_ROOT}")

    prepare_dataset(RAW_ROOT, PROCESSED_ROOT)
    prepare_dataset(RAW_ROOT, PROCESSED_MINI_ROOT, max_images_per_class=MAX_IMAGES_PER_CLASS)

    total_processed = sum(len(list((PROCESSED_ROOT / split).rglob("*.jpg"))) for split in SPLITS)
    total_mini = sum(len(list((PROCESSED_MINI_ROOT / split).rglob("*.jpg"))) for split in SPLITS)
    print(f"Created {PROCESSED_ROOT} with {total_processed} images.")
    print(f"Created {PROCESSED_MINI_ROOT} with {total_mini} images.")


if __name__ == "__main__":
    main()
