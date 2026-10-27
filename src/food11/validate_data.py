from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from food11.data import CLASSES, SPLITS, VALID_SUFFIXES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a processed Food-11 dataset")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)
    return parser.parse_args()


def report(name: str, passed: bool, detail: str = "") -> bool:
    suffix = f" - {detail}" if detail else ""
    print(f"{'PASS' if passed else 'FAIL'}: {name}{suffix}")
    return passed


def image_paths(root: Path, split: str, category: str) -> list[Path]:
    folder = root / split / category
    if not folder.is_dir():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    )


def validate(dataset: Path, min_per_class: int, tau: float) -> bool:
    expected_classes = set(CLASSES)
    split_layout_ok = True
    counts: dict[str, Counter[str]] = {}
    all_images: list[tuple[str, str, Path]] = []

    for split in SPLITS:
        split_dir = dataset / split
        actual_classes = {path.name for path in split_dir.iterdir() if path.is_dir()} if split_dir.is_dir() else set()
        split_layout_ok &= actual_classes == expected_classes
        counts[split] = Counter()
        for category in CLASSES:
            paths = image_paths(dataset, split, category)
            counts[split][category] = len(paths)
            all_images.extend((split, category, path) for path in paths)

    checks = [report("all splits contain exactly the 11 class folders", split_layout_ok)]

    image_errors: list[str] = []
    for _, _, path in all_images:
        try:
            with Image.open(path) as image:
                image.load()
                if image.mode != "RGB" or image.size != (128, 128):
                    image_errors.append(f"{path}: mode={image.mode}, size={image.size}")
        except (OSError, UnidentifiedImageError) as exc:
            image_errors.append(f"{path}: {exc}")
    checks.append(report("all images are readable RGB 128x128 files", not image_errors, f"{len(image_errors)} invalid"))

    small_classes = [
        f"{split}/{category}={counts[split][category]}"
        for split in SPLITS
        for category in CLASSES
        if counts[split][category] < min_per_class
    ]
    checks.append(report("every class meets the minimum image count", not small_classes, f"minimum={min_per_class}"))

    digest_splits: dict[str, set[str]] = {}
    for split, _, path in all_images:
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        digest_splits.setdefault(digest, set()).add(split)
    cross_split_duplicates = sum(1 for splits in digest_splits.values() if len(splits) > 1)
    checks.append(report("no image content appears in multiple splits", cross_split_duplicates == 0, f"duplicates={cross_split_duplicates}"))

    train_total = sum(counts["training"].values())
    validation_total = sum(counts["validation"].values())
    distribution_ok = train_total > 0 and validation_total > 0
    max_delta = 0.0
    if distribution_ok:
        for category in CLASSES:
            train_share = counts["training"][category] / train_total
            validation_share = counts["validation"][category] / validation_total
            max_delta = max(max_delta, abs(train_share - validation_share))
        distribution_ok = max_delta <= tau
    checks.append(report("training and validation class shares match", distribution_ok, f"max_delta={max_delta:.4f}, tau={tau:.4f}"))

    return all(checks)


def main() -> None:
    args = parse_args()
    if not args.dataset.is_dir():
        print(f"FAIL: dataset directory does not exist - {args.dataset}")
        raise SystemExit(1)
    raise SystemExit(0 if validate(args.dataset, args.min_per_class, args.tau) else 1)


if __name__ == "__main__":
    main()
