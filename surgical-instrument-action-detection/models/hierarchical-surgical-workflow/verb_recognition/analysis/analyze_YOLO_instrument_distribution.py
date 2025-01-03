"""
YOLO Dataset Class Distribution Analysis

This script analyzes and visualizes the class distribution in a YOLO format dataset.
It processes label files to generate statistics and create visualization plots.

Usage:
    python3 analyze_distribution.py --labels_path /path/to/labels --yaml_path data.yaml

Features:
    - Processes YOLO format label files (.txt)
    - Generates class distribution statistics
    - Creates visualization plots
    - Supports custom class names
"""

import os
from collections import Counter
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import argparse
import yaml
from pathlib import Path


def analyze_class_distribution(labels_path: str) -> Tuple[Counter, int]:
    """
    Analyze class distribution in YOLO format label files.

    Args:
        labels_path: Path to directory containing YOLO label files

    Returns:
        Tuple containing:
            - Counter object with class counts
            - Total number of objects

    Raises:
        FileNotFoundError: If labels directory doesn't exist
        ValueError: If no valid label files are found
    """
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels directory not found: {labels_path}")

    class_counts = Counter()
    total_objects = 0
    valid_files = False

    for filename in os.listdir(labels_path):
        if filename.endswith('.txt'):
            valid_files = True
            try:
                with open(os.path.join(labels_path, filename), 'r') as f:
                    for line in f:
                        try:
                            class_id = int(line.split()[0])
                            class_counts[class_id] += 1
                            total_objects += 1
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Invalid line in {filename}: {line.strip()}")
                            continue
            except IOError as e:
                print(f"Warning: Could not read file {filename}: {e}")
                continue

    if not valid_files:
        raise ValueError(f"No valid label files found in {labels_path}")

    return class_counts, total_objects


def plot_distribution(
    class_counts: Counter,
    class_names: List[str],
    total_objects: int,
    output_path: str = 'class_distribution.png'
) -> None:
    """
    Create and save a bar plot of class distribution.

    Args:
        class_counts: Counter object with class counts
        class_names: List of class names
        total_objects: Total number of objects
        output_path: Path where to save the plot

    Raises:
        ValueError: If class counts and names don't match
    """
    if len(class_names) < len(class_counts):
        raise ValueError(
            f"More classes in data ({len(class_counts)}) than names provided ({len(class_names)})"
        )

    classes = list(range(len(class_names)))
    counts = [class_counts[i] for i in classes]
    percentages = [count / total_objects * 100 for count in counts]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, percentages)
    plt.xlabel('Class')
    plt.ylabel('Percentage')
    plt.title('Class Distribution in Training Dataset')
    plt.xticks(classes, class_names, rotation=45, ha='right')

    # Add percentage labels on top of bars
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{percentage:.1f}%',
            ha='center',
            va='bottom'
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_statistics(
    class_counts: Counter,
    class_names: List[str],
    total_objects: int
) -> None:
    """
    Print class distribution statistics.

    Args:
        class_counts: Counter object with class counts
        class_names: List of class names
        total_objects: Total number of objects
    """
    print("\nClass Distribution:")
    print("-" * 40)
    for i, name in enumerate(class_names):
        count = class_counts[i]
        percentage = count / total_objects * 100
        print(f"{name:<15}: {count:>6} ({percentage:>6.1f}%)")
    print("-" * 40)
    print(f"Total objects: {total_objects}\n")


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(
        description='Analyze class distribution in YOLO dataset'
    )
    parser.add_argument(
        '--labels_path',
        type=str,
        default='labels/train',
        help='Path to YOLO format label files'
    )
    parser.add_argument(
        '--yaml_path',
        type=str,
        help='Path to data.yaml file containing class names'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='class_distribution.png',
        help='Output path for distribution plot'
    )
    args = parser.parse_args()

    # Read class names from YAML if provided, otherwise use defaults
    if args.yaml_path:
        with open(args.yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            class_names = data.get('names', [])
    else:
        class_names = [
            'Grasper', 'Bipolar', 'Hook', 'Scissors',
            'Clipper', 'Irrigator', 'SpecimenBag'
        ]

    try:
        # Analyze distribution
        class_counts, total_objects = analyze_class_distribution(args.labels_path)

        # Print statistics
        print_statistics(class_counts, class_names, total_objects)

        # Create visualization
        plot_distribution(class_counts, class_names, total_objects, args.output)
        print(f"Distribution plot saved as '{args.output}'")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())