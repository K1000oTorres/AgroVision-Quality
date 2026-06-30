from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any
import hashlib
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def safe_stem(original_stem: str, max_length: int = 80) -> str:
    if len(original_stem) <= max_length:
        return original_stem

    digest = hashlib.md5(original_stem.encode("utf-8")).hexdigest()[:12]
    return f"{original_stem[:max_length]}_{digest}"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if "names" not in data or "nc" not in data:
        raise ValueError("data.yaml must contain 'names' and 'nc'.")

    if int(data["nc"]) != len(data["names"]):
        raise ValueError("'nc' does not match number of class names.")

    return data


def find_image_path(images_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def parse_detection(parts: list[str], num_classes: int) -> str | None:
    if len(parts) != 5:
        return None

    try:
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])
    except ValueError:
        return None

    if class_id < 0 or class_id >= num_classes:
        return None

    x_center = clamp(x_center)
    y_center = clamp(y_center)
    width = clamp(width)
    height = clamp(height)

    if width <= 0 or height <= 0:
        return None

    return f"{class_id} {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}"


def parse_segmentation(parts: list[str], num_classes: int) -> str | None:
    if len(parts) < 7:
        return None

    if (len(parts) - 1) % 2 != 0:
        return None

    try:
        class_id = int(parts[0])
        coordinates = list(map(float, parts[1:]))
    except ValueError:
        return None

    if class_id < 0 or class_id >= num_classes:
        return None

    x_values = [clamp(value) for value in coordinates[0::2]]
    y_values = [clamp(value) for value in coordinates[1::2]]

    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)

    width = x_max - x_min
    height = y_max - y_min

    if width <= 0 or height <= 0:
        return None

    x_center = x_min + width / 2
    y_center = y_min + height / 2

    return f"{class_id} {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}"

def convert_label_file(
    source_label_path: Path,
    target_label_path: Path,
    num_classes: int,
) -> tuple[int, int, int]:
    detection_rows = 0
    segmentation_rows = 0
    discarded_rows = 0

    converted_lines: list[str] = []

    lines = source_label_path.read_text(encoding="utf-8").splitlines()

    for line in lines:
        parts = line.strip().split()

        if not parts:
            discarded_rows += 1
            continue

        detection_line = parse_detection(parts, num_classes)

        if detection_line is not None:
            converted_lines.append(detection_line)
            detection_rows += 1
            continue

        segmentation_line = parse_segmentation(parts, num_classes)

        if segmentation_line is not None:
            converted_lines.append(segmentation_line)
            segmentation_rows += 1
            continue

        discarded_rows += 1
    target_label_path.parent.mkdir(parents=True, exist_ok=True)

    if converted_lines:
        target_label_path.write_text(
            "\n".join(converted_lines) + "\n",
            encoding="utf-8",
        )

    return detection_rows, segmentation_rows, discarded_rows


def process_split(
    split_name: str,
    source_root: Path,
    target_root: Path,
    num_classes: int,
) -> dict[str, int]:
    source_images_dir = source_root / split_name / "images"
    source_labels_dir = source_root / split_name / "labels"

    target_images_dir = target_root / split_name / "images"
    target_labels_dir = target_root / split_name / "labels"

    target_images_dir.mkdir(parents=True, exist_ok=True)
    target_labels_dir.mkdir(parents=True, exist_ok=True)

    if not source_images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {source_images_dir}")

    if not source_labels_dir.exists():
        raise FileNotFoundError(f"Missing labels directory: {source_labels_dir}")

    target_images_dir.mkdir(parents=True, exist_ok=True)
    target_labels_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "images_copied": 0,
        "images_skipped": 0,
        "label_files_written": 0,
        "detection_rows": 0,
        "segmentation_rows_converted": 0,
        "discarded_rows": 0,
        "empty_or_invalid_label_files": 0,
    }

    for source_label_path in sorted(source_labels_dir.glob("*.txt")):
        image_path = find_image_path(source_images_dir, source_label_path.stem)

        if image_path is None:
            stats["images_skipped"] += 1
            continue

        short_stem = safe_stem(source_label_path.stem)
        target_label_path = target_labels_dir / f"{short_stem}.txt"

        detection_rows, segmentation_rows, discarded_rows = convert_label_file(
            source_label_path=source_label_path,
            target_label_path=target_label_path,
            num_classes=num_classes,
        )

        stats["detection_rows"] += detection_rows
        stats["segmentation_rows_converted"] += segmentation_rows
        stats["discarded_rows"] += discarded_rows

        if not target_label_path.exists():
            stats["empty_or_invalid_label_files"] += 1
            continue

        target_image_path = target_images_dir / f"{short_stem}{image_path.suffix.lower()}"
        shutil.copy2(image_path, target_image_path)
        stats["images_copied"] += 1
        stats["label_files_written"] += 1

    return stats


def write_processed_yaml(
    source_yaml: dict[str, Any],
    target_root: Path,
) -> None:
    processed_yaml = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": source_yaml["nc"],
        "names": source_yaml["names"],
    }

    with (target_root / "data.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            processed_yaml,
            file,
            sort_keys=False,
            allow_unicode=True,
        )


def write_report(target_root: Path, split_stats: dict[str, dict[str, int]]) -> None:
    lines = [
        "# Processed YOLO Detection Dataset Report",
        "",
        "This dataset was generated from the original dataset without modifying it.",
        "Segmentation polygons were converted to YOLO detection bounding boxes.",
        "",
        "| Split | Images copied | Labels written | Detection rows | Segmentation rows converted | Discarded rows | Empty/invalid label files |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for split_name, stats in split_stats.items():
        lines.append(
            f"| {split_name} | "
            f"{stats['images_copied']} | "
            f"{stats['label_files_written']} | "
            f"{stats['detection_rows']} | "
            f"{stats['segmentation_rows_converted']} | "
            f"{stats['discarded_rows']} | "
            f"{stats['empty_or_invalid_label_files']} |"
        )

    (target_root / "processing_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def prepare_dataset(source_root: Path, target_root: Path) -> None:
    source_yaml_path = source_root / "data.yaml"

    if target_root.exists():
        raise FileExistsError(
            f"Target directory already exists: {target_root}. "
            "Remove it manually or choose another output path."
        )

    source_yaml = load_yaml(source_yaml_path)
    num_classes = int(source_yaml["nc"])

    split_stats: dict[str, dict[str, int]] = {}

    for split_name in ["train", "valid", "test"]:
        split_stats[split_name] = process_split(
            split_name=split_name,
            source_root=source_root,
            target_root=target_root,
            num_classes=num_classes,
        )

    write_processed_yaml(source_yaml, target_root)
    write_report(target_root, split_stats)

    print("Dataset processed successfully.")
    print(f"Output directory: {target_root}")
    print(f"Processed data.yaml: {target_root / 'data.yaml'}")
    print(f"Processing report: {target_root / 'processing_report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a mixed YOLO detection/segmentation dataset into a "
            "YOLO detection dataset."
        )
    )

    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Original dataset root directory containing data.yaml.",
    )

    parser.add_argument(
        "--target-root",
        type=Path,
        required=True,
        help="Output directory for the processed YOLO detection dataset.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prepare_dataset(
        source_root=args.source_root.resolve(),
        target_root=args.target_root.resolve(),
    )


if __name__ == "__main__":
    main()