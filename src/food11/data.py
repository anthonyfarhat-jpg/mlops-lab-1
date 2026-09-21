from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

CATEGORY_NAMES = [
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

RAW_ROOT = Path("data/food11_raw")
PROCESSED_ROOT = Path("data/food11_processed")
PROCESSED_MINI_ROOT = Path("data/food11_processed_mini")
MAX_IMAGES_PER_CLASS = 100
SPLITS = ("training", "evaluation", "validation")
VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}


def label_to_category(file_name: str) -> str:
    label = file_name.split("_")[0]
    index = int(label)
    if not 0 <= index < len(CATEGORY_NAMES):
        raise ValueError(f"Unexpected label {label!r} in filename {file_name!r}")
    return CATEGORY_NAMES[index]


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

    for split in SPLITS:
        split_source = source_root / split
        if not split_source.exists():
            continue

        for category in CATEGORY_NAMES:
            category_files = collect_class_files(split_source)
            files = category_files[category]
            if max_images_per_class is not None:
                files = files[:max_images_per_class]

            target_dir = destination_root / split / category
            target_dir.mkdir(parents=True, exist_ok=True)

            for image_path in files:
                with Image.open(image_path) as image:
                    rgb_image = image.convert("RGB")
                    resized = rgb_image.resize((128, 128), Image.Resampling.BILINEAR)
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
