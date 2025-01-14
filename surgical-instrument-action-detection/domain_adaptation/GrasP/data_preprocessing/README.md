# GraSP Dataset Preprocessing Pipeline

This repository contains scripts to preprocess the GraSP (Graphical Surgical Phase) dataset. The preprocessing includes restructuring the original 30 FPS annotations into a more comprehensive format at 1 FPS, including detailed instrument information, actions, and localization data.

## Initial Setup

1. **Download the Original Dataset**
   - Access the GraSP dataset from [Google Drive](https://drive.google.com/drive/folders/16uGgYsQ2oohKo1-iSxOFWnFAPlGTtvb9)
   - Download all CASExx folders and annotation files
   - You need both training and test data:
     - `grasp_short-term_train.json`
     - `grasp_short-term_test.json`
     - All CASE* folders

2. **Set Up Directory Structure**
   - Create your working directory with the following structure:
   ```
   working_directory/
   ├── scripts/
   │   ├── reorganize_grasp_json.py
   │   └── create_video_folders.py
   ├── config.json
   └── GraSP/
       ├── 30fps/
       │   ├── train/
       │   │   ├── grasp_short-term_train.json
       │   │   └── CASE*/               # Place downloaded CASE folders here
       │   └── test/
       │       ├── grasp_short-term_test.json
       │       └── CASE*/               # Place downloaded CASE folders here
       └── 1fps/                        # Will be created by scripts
           ├── train/
           └── test/

   ```

3. **Place Downloaded Files**
   - Move training CASE folders (CASE001,CASE002,CASE004, CASE004, CASE007, CASE014, CASE015, CASE021) to `GraSP/30fps/train/`
   - Move all test CASE folders (CASE041,CASE047,CASE050, CASE051, CASE053) to `GraSP/30fps/test/`
   - Place `grasp_short-term_train.json` in `GraSP/30fps/train/`
   - Place `grasp_short-term_test.json` in `GraSP/30fps/test/`

## Introduction
This repository contains the preprocessing pipeline for the GraSP (Multi-Granular Surgical Scene Understanding of Prostatectomies) dataset. The original dataset at 30 FPS provides limited annotations, containing only phase and step IDs. For comprehensive model evaluation in instrument and action recognition, a pipeline was developed to process these annotations to access detailed instrument information, actions, and localization data.

## Preprocessing Pipeline

### 1. JSON Restructuring
The original annotations in `grasp_short-term_train.json` and `grasp_short-term_test.json` were reorganized into a more hierarchical structure to:
- Consolidate all frame-specific information in one location
- Add missing instrument and action annotations
- Improve data accessibility and processing efficiency

### Data Structure
The restructured JSON format follows this schema:
```json
{
    "video_id": "VIDxx",
    "original_case": "CASExx",
    "metadata": {
        "width": 1280,
        "height": 800,
        "fps": 1
    },
    "categories": {
        "instruments": [...],
        "actions": [...],
        "phases": [...],
        "steps": [...]
    },
    "frames": {
        "frame_xxx.jpg": {
            "frame_num": int,
            "file_name": "frame_xxx.jpg",
            "phase": int,
            "step": int,
            "instruments": [
                {
                    "id": int,
                    "category_id": int,
                    "area": float,
                    "bbox": [...],
                    "segmentation": {...},
                    "actions": [...],
                    "iscrowd": int
                }
            ]
        }
    }
}
```

## Usage

### Prerequisites
- Python 3.7+
- Required packages: json, os, shutil, pathlib


#### Step 1: JSON Restructuring
Run the first script to reorganize the JSON annotations:
```bash
python reorganize_grasp_json.py
```

#### Step 2: Image Extraction
Run the second script to extract and organize the corresponding images:
```bash
python create_video_folders.py
```

## Implementation Details

### JSON Restructuring Script
The `reorganize_grasp_json.py` script processes the original JSON annotations to create a more structured format. It:
1. Reads the original annotation file
2. Creates a new hierarchical structure
3. Groups annotations by frame
4. Organizes instrument and action information
5. Saves the restructured JSON

### Image Organization Script
The `create_video_folders.py` script manages the image data by:
1. Reading the restructured JSON files
2. Creating appropriate folder structures
3. Copying relevant frames from the 30 FPS dataset
4. Organizing them according to the new structure

## Notes
- Original 30 FPS annotations only contain phase and step IDs
- Restructured format includes comprehensive information about instruments, actions, and localization data
- Processed dataset maintains 1 FPS sampling rate for efficiency
- All original image dimensions and quality are preserved

## Final Dataset Structure
The processed dataset maintains the following hierarchical structure:
```
GraSP/
├── 30fps/
│   ├── test/
│   │   ├── Videos/
│   │   └── Labels/
│   └── train/
│       ├── Videos/
│       └── Labels/
└── 1fps/
    ├── test/
    │   ├── Videos/
    │   └── Labels/
    └── train/
        ├── Videos/
        └── Labels/
```
