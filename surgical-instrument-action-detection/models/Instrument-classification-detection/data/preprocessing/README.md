# Data Preprocessing
## Table of Contents

Overview
VOC to YOLO Conversion

Dataset Source
Prerequisites
Directory Structure
Usage
Output
Workflow Context


Troubleshooting

## Overview
This directory contains tools for preprocessing the surgical instrument dataset, specifically for converting annotations from VOC format to YOLO format. This conversion is necessary to prepare the data for our active learning pipeline.
## VOC to YOLO Conversion
### Dataset Source
The initial dataset and annotations are sourced from Jin et al. (2018):
```bibtex
@inproceedings{jin2018tool,
title={Tool detection and operative skill assessment in surgical videos using region-based convolutional neural networks},
author={Jin, Amy and Yeung, Serena and Jopling, Jeffrey and Krause, Jonathan and Azagury, Dan and Milstein, Arnold and Fei-Fei, Li},
booktitle={2018 IEEE winter conference on applications of computer vision (WACV)},
pages={691--699},
year={2018},
organization={IEEE}
}
```
### Prerequisites

Python 3.8+
Required packages:
```bash
pip install pyyaml
```

### Directory Structure
Expected input directory structure from the original dataset:
```
m2cai16-tool-locations/
├── Annotations/     # XML files in VOC format
├── JPEGImages/     # Image files from surgical videos
├── ImageSets/
│   └── Main/       # Contains train.txt, val.txt, etc.
└── class_list.txt  # List of surgical instruments
```
Output directory structure after conversion:
```
yolo_dataset/
├── images/         # Copied image files
├── labels/         # Converted YOLO format annotations
├── train.txt       # Training set file paths
├── val.txt        # Validation set file paths
├── test.txt       # Test set file paths
└── data.yaml      # Dataset configuration for YOLO
```
### Usage

Ensure you have the 'm2cai16-tool-locations' directory from Jin et al. (2018)
Run the conversion script:
```bash
python voc_to_yolo_converter.py 
--input-dir /path/to/m2cai16-tool-locations 
--output-dir /path/to/yolo_dataset
```
Example with specific paths:
```bash
python voc_to_yolo_converter.py 
--input-dir /Users/username/Desktop/m2cai16-tool-locations 
--output-dir /Users/username/Desktop/yolo_dataset
```

### Output
The script generates:

Converted annotation files in YOLO format
Copied image files in the new structure
Dataset split files (train.txt, val.txt, test.txt)
Configuration file (data.yaml) containing:

File paths
Number of classes
Class names



### Workflow Context
This conversion is Step 1 in our active learning pipeline:

**Current Step**: Convert VOC annotations to YOLO format
**Next Steps**:

Train initial YOLO model
Generate predictions on new video frames
Begin active learning annotation refinement



## Troubleshooting
Common issues and solutions:

**Missing class_list.txt**

Ensure the file exists in the root of m2cai16-tool-locations
Verify file format: one class per line with index and name


**File Path Issues**

Use absolute paths if relative paths fail
Check for spaces or special characters in paths


**Conversion Errors**

Check the conversion log file in the output directory
Verify XML files are properly formatted
Ensure all referenced images exist
