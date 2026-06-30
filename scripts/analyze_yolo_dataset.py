from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import yaml


@dataclass(frozen=True)
class DatasetPaths:
    root_dir: Path
    yaml_path: Path
    train_images: Path
    train_labels: Path
    valid_images: Path
    valid_labels: Path
    test_images: Path
    test_labels: Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(yaml_path: Path) -> dict[str, Any]:
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("Invalid data.yaml format.")

    if "names" not in data or "nc" not in data:
        raise ValueError("data.yaml must contain 'names' and 'nc'.")

    if int(data["nc"]) != len(data["names"]):
        raise ValueError("Mismatch between 'nc' and number of class names.")

    return data


def resolve_dataset_paths(yaml_path: Path) -> DatasetPaths:
    root_dir = yaml_path.parent

    return DatasetPaths(
        root_dir=root_dir,
        yaml_path=yaml_path,
        train_images=(root_dir / "train" / "images").resolve(),
        train_labels=(root_dir / "train" / "labels").resolve(),
        valid_images=(root_dir / "valid" / "images").resolve(),
        valid_labels=(root_dir / "valid" / "labels").resolve(),
        test_images=(root_dir / "test" / "images").resolve(),
        test_labels=(root_dir / "test" / "labels").resolve(),
    )


def validate_directories(paths: DatasetPaths) -> None:
    required_dirs = [
        paths.train_images,
        paths.train_labels,
        paths.valid_images,
        paths.valid_labels,
        paths.test_images,
        paths.test_labels,
    ]

    missing_dirs = [path for path in required_dirs if not path.exists()]

    if missing_dirs:
        missing_text = "\n".join(str(path) for path in missing_dirs)
        raise FileNotFoundError(f"Missing required directories:\n{missing_text}")


def list_images(images_dir: Path) -> list[Path]:
    return sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def label_path_for_image(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def parse_label_file(
    label_path: Path,
    class_names: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    annotations: list[dict[str, Any]] = []
    errors: list[str] = []

    if not label_path.exists():
        return annotations, [f"Missing label file: {label_path}"]

    lines = label_path.read_text(encoding="utf-8").splitlines()

    if not lines:
        return annotations, [f"Empty label file: {label_path}"]

    for line_number, line in enumerate(lines, start=1):
        parts = line.strip().split()

        if len(parts) != 5:
            errors.append(
                f"Invalid YOLO row at {label_path}:{line_number} -> {line}"
            )
            continue

        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            errors.append(
                f"Invalid numeric values at {label_path}:{line_number} -> {line}"
            )
            continue

        if class_id < 0 or class_id >= len(class_names):
            errors.append(
                f"Invalid class_id {class_id} at {label_path}:{line_number}"
            )
            continue

        values = [x_center, y_center, width, height]
        if any(value < 0 or value > 1 for value in values):
            errors.append(
                f"Bounding box values out of range at "
                f"{label_path}:{line_number} -> {line}"
            )
            continue

        if width <= 0 or height <= 0:
            errors.append(
                f"Invalid bbox size at {label_path}:{line_number} -> {line}"
            )
            continue

        annotations.append(
            {
                "label_file": str(label_path),
                "class_id": class_id,
                "class_name": class_names[class_id],
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
            }
        )

    return annotations, errors


def analyze_split(
    split_name: str,
    images_dir: Path,
    labels_dir: Path,
    class_names: list[str],
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []

    images = list_images(images_dir)
    image_stems = {image.stem for image in images}
    label_files = sorted(labels_dir.glob("*.txt"))
    label_stems = {label.stem for label in label_files}

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    for stem in sorted(missing_labels):
        errors.append(f"[{split_name}] Missing label for image: {stem}")

    for stem in sorted(orphan_labels):
        errors.append(f"[{split_name}] Orphan label without image: {stem}")

    for image_path in images:
        label_path = label_path_for_image(image_path, labels_dir)
        annotations, label_errors = parse_label_file(label_path, class_names)

        errors.extend(f"[{split_name}] {error}" for error in label_errors)

        for annotation in annotations:
            records.append(
                {
                    "split": split_name,
                    "image_file": str(image_path),
                    **annotation,
                }
            )

    stats = {
        "images": len(images),
        "label_files": len(label_files),
        "missing_labels": len(missing_labels),
        "orphan_labels": len(orphan_labels),
        "annotations": len(records),
    }

    return records, errors, stats


def split_fruit_and_ripeness(class_name: str) -> tuple[str, str]:
    parts = class_name.split()

    if len(parts) < 2:
        return class_name.lower(), "unknown"

    fruit = parts[0].lower()
    ripeness = parts[-1].lower()

    return fruit, ripeness


def create_summary_tables(
    annotations_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    class_distribution = (
        annotations_df.groupby(["class_id", "class_name"])
        .size()
        .reset_index(name="annotations")
        .sort_values("annotations", ascending=False)
    )

    fruit_distribution = (
        annotations_df.groupby("fruit_type")
        .size()
        .reset_index(name="annotations")
        .sort_values("annotations", ascending=False)
    )

    ripeness_distribution = (
        annotations_df.groupby("ripeness")
        .size()
        .reset_index(name="annotations")
        .sort_values("annotations", ascending=False)
    )

    split_distribution = (
        annotations_df.groupby("split")
        .size()
        .reset_index(name="annotations")
    )

    class_distribution.to_csv(output_dir / "class_distribution.csv", index=False)
    fruit_distribution.to_csv(output_dir / "fruit_distribution.csv", index=False)
    ripeness_distribution.to_csv(
        output_dir / "ripeness_distribution.csv",
        index=False,
    )
    split_distribution.to_csv(output_dir / "split_distribution.csv", index=False)


def plot_distribution(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(14, 8))
    plt.bar(data[x_column], data[y_column])
    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def create_plots(annotations_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    class_distribution = (
        annotations_df.groupby("class_name")
        .size()
        .reset_index(name="annotations")
        .sort_values("annotations", ascending=False)
    )

    fruit_distribution = (
        annotations_df.groupby("fruit_type")
        .size()
        .reset_index(name="annotations")
        .sort_values("annotations", ascending=False)
    )

    ripeness_distribution = (
        annotations_df.groupby("ripeness")
        .size()
        .reset_index(name="annotations")
        .sort_values("annotations", ascending=False)
    )

    plot_distribution(
        class_distribution,
        "class_name",
        "annotations",
        "Class distribution",
        output_dir / "class_distribution.png",
    )

    plot_distribution(
        fruit_distribution,
        "fruit_type",
        "annotations",
        "Fruit distribution",
        output_dir / "fruit_distribution.png",
    )

    plot_distribution(
        ripeness_distribution,
        "ripeness",
        "annotations",
        "Ripeness distribution",
        output_dir / "ripeness_distribution.png",
    )


def draw_yolo_bbox(
    image: Any,
    annotation: pd.Series,
    class_name: str,
) -> None:
    height, width = image.shape[:2]

    x_center = float(annotation["x_center"]) * width
    y_center = float(annotation["y_center"]) * height
    bbox_width = float(annotation["width"]) * width
    bbox_height = float(annotation["height"]) * height

    x1 = int(x_center - bbox_width / 2)
    y1 = int(y_center - bbox_height / 2)
    x2 = int(x_center + bbox_width / 2)
    y2 = int(y_center + bbox_height / 2)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        image,
        class_name,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def create_annotated_samples(
    annotations_df: pd.DataFrame,
    output_dir: Path,
    samples_per_split: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in sorted(annotations_df["split"].unique()):
        split_df = annotations_df[annotations_df["split"] == split_name]
        image_files = sorted(split_df["image_file"].unique())

        selected_images = random.sample(
            image_files,
            k=min(samples_per_split, len(image_files)),
        )

        for image_file in selected_images:
            image_path = Path(image_file)
            image = cv2.imread(str(image_path))

            if image is None:
                continue

            image_annotations = split_df[split_df["image_file"] == image_file]

            for _, annotation in image_annotations.iterrows():
                draw_yolo_bbox(
                    image=image,
                    annotation=annotation,
                    class_name=str(annotation["class_name"]),
                )

            output_path = output_dir / f"{split_name}_{image_path.name}"
            cv2.imwrite(str(output_path), image)


def write_markdown_report(
    output_dir: Path,
    dataset_stats: dict[str, dict[str, int]],
    annotations_df: pd.DataFrame,
    errors: list[str],
) -> None:
    total_images = sum(stats["images"] for stats in dataset_stats.values())
    total_annotations = len(annotations_df)

    class_counter = Counter(annotations_df["class_name"])
    fruit_counter = Counter(annotations_df["fruit_type"])
    ripeness_counter = Counter(annotations_df["ripeness"])

    most_common_class = class_counter.most_common(1)[0]
    least_common_class = class_counter.most_common()[-1]

    report = [
        "# YOLO Dataset Analysis Report",
        "",
        "## General Summary",
        "",
        f"- Total images: {total_images}",
        f"- Total annotations: {total_annotations}",
        f"- Total classes: {annotations_df['class_name'].nunique()}",
        f"- Total fruit types: {annotations_df['fruit_type'].nunique()}",
        f"- Total ripeness states: {annotations_df['ripeness'].nunique()}",
        f"- Detected validation errors: {len(errors)}",
        "",
        "## Split Summary",
        "",
        "| Split | Images | Label files | Annotations | Missing labels | Orphan labels |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for split_name, stats in dataset_stats.items():
        report.append(
            f"| {split_name} | {stats['images']} | {stats['label_files']} | "
            f"{stats['annotations']} | {stats['missing_labels']} | "
            f"{stats['orphan_labels']} |"
        )

    report.extend(
        [
            "",
            "## Class Balance",
            "",
            f"- Most common class: {most_common_class[0]} "
            f"({most_common_class[1]} annotations)",
            f"- Least common class: {least_common_class[0]} "
            f"({least_common_class[1]} annotations)",
            "",
            "## Fruit Distribution",
            "",
            "| Fruit | Annotations |",
            "|---|---:|",
        ]
    )

    for fruit, count in fruit_counter.most_common():
        report.append(f"| {fruit} | {count} |")

    report.extend(
        [
            "",
            "## Ripeness Distribution",
            "",
            "| Ripeness | Annotations |",
            "|---|---:|",
        ]
    )

    for ripeness, count in ripeness_counter.most_common():
        report.append(f"| {ripeness} | {count} |")

    if errors:
        report.extend(
            [
                "",
                "## Validation Errors",
                "",
                "The following issues were detected:",
                "",
            ]
        )

        for error in errors[:200]:
            report.append(f"- {error}")

        if len(errors) > 200:
            report.append(f"- ... and {len(errors) - 200} more errors.")

    (output_dir / "dataset_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )


def analyze_dataset(yaml_path: Path, output_dir: Path, samples_per_split: int) -> None:
    data_yaml = load_yaml(yaml_path)
    class_names = list(data_yaml["names"])
    paths = resolve_dataset_paths(yaml_path)

    validate_directories(paths)

    all_records: list[dict[str, Any]] = []
    all_errors: list[str] = []
    dataset_stats: dict[str, dict[str, int]] = {}

    split_configs = {
        "train": (paths.train_images, paths.train_labels),
        "valid": (paths.valid_images, paths.valid_labels),
        "test": (paths.test_images, paths.test_labels),
    }

    for split_name, (images_dir, labels_dir) in split_configs.items():
        records, errors, stats = analyze_split(
            split_name=split_name,
            images_dir=images_dir,
            labels_dir=labels_dir,
            class_names=class_names,
        )

        all_records.extend(records)
        all_errors.extend(errors)
        dataset_stats[split_name] = stats

    if not all_records:
        raise ValueError("No valid annotations found in the dataset.")

    annotations_df = pd.DataFrame(all_records)

    fruit_and_ripeness = annotations_df["class_name"].apply(
        split_fruit_and_ripeness,
    )

    annotations_df["fruit_type"] = [
        value[0] for value in fruit_and_ripeness
    ]
    annotations_df["ripeness"] = [
        value[1] for value in fruit_and_ripeness
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    annotations_df.to_csv(output_dir / "annotations.csv", index=False)

    create_summary_tables(annotations_df, output_dir)
    create_plots(annotations_df, output_dir / "plots")
    create_annotated_samples(
        annotations_df=annotations_df,
        output_dir=output_dir / "samples",
        samples_per_split=samples_per_split,
    )
    write_markdown_report(
        output_dir=output_dir,
        dataset_stats=dataset_stats,
        annotations_df=annotations_df,
        errors=all_errors,
    )

    print("Dataset analysis completed successfully.")
    print(f"Report: {output_dir / 'dataset_report.md'}")
    print(f"Annotations CSV: {output_dir / 'annotations.csv'}")
    print(f"Plots directory: {output_dir / 'plots'}")
    print(f"Samples directory: {output_dir / 'samples'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a YOLO object detection dataset.",
    )

    parser.add_argument(
        "--data-yaml",
        type=Path,
        required=True,
        help="Path to data.yaml.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/dataset_analysis"),
        help="Directory where the analysis report will be saved.",
    )

    parser.add_argument(
        "--samples-per-split",
        type=int,
        default=10,
        help="Number of annotated sample images to generate per split.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        analyze_dataset(
            yaml_path=args.data_yaml.resolve(),
            output_dir=args.output_dir.resolve(),
            samples_per_split=args.samples_per_split,
        )
    except Exception as exc:
        raise RuntimeError(f"Dataset analysis failed: {exc}") from exc


if __name__ == "__main__":
    main()
    