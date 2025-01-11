# HeiChole Dataset Processing

## Dataset Access

The HeiChole dataset is available at: [HeiChole on Synapse](https://www.synapse.org/Synapse:syn18824884/wiki/592580)

Dataset access steps:
1. Register for a Synapse account
2. Request access to the dataset
3. Navigate to "Files" → "Videos" → "Full"
4. Download the desired video files

## Dependencies

- Python 3.7+
- OpenCV (cv2)
- tqdm
- pandas
- pathlib

## Video Processing

### Usage

\```bash
python video_preprocessing.py /path/to/video /path/to/output
\```

### Advanced Options
\```bash
python video_preprocessing.py \
    /path/to/video \
    /path/to/output \
    --fps 1.0 \
    --size 512 512 \
    --format png
\```

### Parameters
\```bash
- `--fps`: target frame rate for extraction (default: 1.0)
- `--size`: target frame size as width height (default: 512 512)
- `--format`: output image format (png or jpg, default: png)
\```

### Storage Information
Storage requirements for processing:
- Example for one video (807.78 MB MP4):
  - 54,930 frames at full frame rate
  - ~34.92 GB extracted frames
  - Ratio: 1 MB MP4 ≈ 68 frames ≈ 43.23 MB extracted frames

The implemented solution reduces storage requirements through:
- Extraction with reduced frame rate (default: 1 fps)
- Frame size adjustment (default: 512x512)
- Optional jpg compression

## Label Processing

### Overview

The label processing script (`label_preprocessing.py`) aligns the original HeiChole dataset labels with the extracted video frames. It processes three types of annotations:
- Surgical phases (7 classes)
- Instrument presence (7 instruments)
- Surgical actions (4 types)

### Usage

\```bash
python label_preprocessing.py /path/to/heichole/dataset /path/to/output
\```

### Features

- Combines phase, instrument, and action labels into a single JSON structure
- Samples labels at intervals matching extracted video frames
- Provides consistent label mapping and structure
- Handles multiple video groups with different sampling rates

### Label Types

1. Surgical Phases:
   - Preparation
   - Calot triangle dissection
   - Clipping and cutting
   - Gallbladder dissection
   - Gallbladder packaging
   - Cleaning and coagulation
   - Gallbladder retraction

2. Instruments:
   - Grasper
   - Clipper
   - Coagulation
   - Scissors
   - Suction irrigation
   - Specimen bag
   - Stapler

3. Actions:
   - Grasp
   - Hold
   - Cut
   - Clip

### Output Format

The script generates JSON files with the following structure:
\```json
{
  "video_id": "Hei-CholeXX",
  "frames": {
    "frame_number": {
      "phase": {
        "id": 0,
        "name": "Preparation"
      },
      "instruments": {
        "grasper": 1,
        "clipper": 0,
        ...
      },
      "actions": {
        "grasp": 1,
        "hold": 0,
        ...
      }
    }
  },
  "total_frames": 1000,
  "sample_interval": 25,
  "label_mapping": {
    "phases": {...},
    "instruments": {...},
    "actions": {...}
  }
}
\```

### Storage Information

- Original label files: CSV format
- Processed labels: JSON format
- Approximate size: ~100KB per video (varies with video length and sampling rate)