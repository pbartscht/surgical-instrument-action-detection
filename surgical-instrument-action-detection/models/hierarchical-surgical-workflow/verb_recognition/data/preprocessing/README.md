# Surgical Instrument Verb Training Data Preparation Pipeline

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)

This repository contains a preprocessing pipeline for creating training data for surgical verb recognition as part of a hierarchical deep learning model. The pipeline processes video frames from the CholecT50 dataset, using pre-trained YOLO models for instrument detection to create cropped images for verb training.

## Workflow Overview

1. Instrument Detection: Uses a pre-trained YOLO model to detect surgical instruments in video frames
2. Ground Truth Verification: Matches detections with CholecT50 dataset annotations
3. Data Cleaning: Applies strict filtering rules to ensure high-quality training data
4. Image Processing: Crops and resizes detected instrument regions
5. Label Generation: Creates clean CSV files with verified verb labels

## Prerequisites

### System Requirements
- Python 3.7+
- Required packages:
  ```
  ultralytics
  pytorch
  opencv-python
  pandas
  numpy
  Pillow
  tqdm
  ```
- Pre-trained YOLO model weights (`best.pt`)
- CholecT50 dataset with the following structure:
  ```
  dataset_dir/
  └── CholecT50/
      ├── videos/
      │   ├── VID01/
      │   ├── VID02/
      │   └── ...
      └── labels/
          ├── VID01.json
          ├── VID02.json
          └── ...
  ```

## Installation

1. Clone this repository
2. Install required packages:
   ```bash
   pip install ultralytics torch opencv-python pandas numpy Pillow tqdm
   ```

## Usage

1. Update the path variables in the script:
   ```python
   dataset_dir = "/path/to/your/dataset"
   verbs_dir = os.path.join(dataset_dir, "Verbs")
   model = YOLO('/path/to/your/best.pt')
   ```

2. Run the script:
   ```bash
   python process_surgical_data.py
   ```

## Data Cleaning Rules

The pipeline implements several important cleaning rules to ensure high-quality training data:

1. **Single Instance Rule**: Only processes frames where exactly one instance of an instrument is detected
2. **Unique Verb Rule**: Only includes cases where there is exactly one verb associated with an instrument
3. **Confidence Threshold**: Applies a confidence threshold of 0.6 for YOLO detections
4. **IOU Threshold**: Uses an IOU threshold of 0.3 for non-maximum suppression
5. **Output Standardization**: All cropped images are resized to 256x256 pixels

## Output Structure

The pipeline creates the following directory structure:
```
Verbs/
├── labels/
│   ├── VID01.json
│   ├── VID02.json
│   └── ...
├── VID01/
│   ├── 0001_grasper_grasp_conf0.95.png
│   ├── ...
│   └── labels.csv
├── VID02/
└── ...
```

Each video directory contains:
- Cropped instrument images (256x256 pixels)
- A labels.csv file with columns:
  - Dateiname (Filename)
  - Verb
  - Instrument
  - Frame
  - Confidence

## Supported Instruments and Verbs

### Instruments
- Grasper
- Bipolar
- Hook
- Scissors
- Clipper
- Irrigator

### Verbs
- Grasp
- Retract
- Dissect
- Coagulate
- Clip
- Cut
- Aspirate
- Irrigate
- Pack
- Null_verb

## Notes

- The pipeline processes 45 videos from the CholecT50 dataset
- Only frames with unambiguous instrument-verb pairs are included
- Multiple instruments in a single frame are excluded to prevent incorrect verb assignments
- All image crops maintain aspect ratio before resizing

## Error Handling

The pipeline includes comprehensive error handling:
- Continues processing if individual frames fail
- Logs all processing errors
- Provides progress bars for monitoring
- Generates summary statistics after processing each video